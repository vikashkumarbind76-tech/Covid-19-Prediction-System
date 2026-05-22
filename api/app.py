from functools import lru_cache
import csv, os, io, sqlite3, hashlib, base64
from datetime import datetime
import numpy as np
import pandas as pd
from flask import (
    Flask, Response, jsonify, redirect,
    render_template, request, session, url_for
)

# matplotlib is imported lazily (only when dashboard charts are generated)
# to avoid Vercel cold-start crashes from display/font init
def _get_matplotlib():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    return plt, mpatches

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(BASE_DIR)

# ── Supabase / PostgreSQL support ──────────────────────────────────────────
# Set DATABASE_URL env var to your Supabase connection string to enable Postgres.
# Without it, the app falls back to local SQLite (great for development).
DATABASE_URL = os.environ.get("DATABASE_URL")  # e.g. postgresql://postgres:pass@db.xxxx.supabase.co:5432/postgres
USE_POSTGRES = bool(DATABASE_URL)
if USE_POSTGRES:
    try:
        import psycopg2
        import psycopg2.extras
    except ImportError:
        USE_POSTGRES = False  # psycopg2 not installed → fall back to SQLite

DB_PATH    = os.environ.get("DB_PATH", "/tmp/predictions.db" if os.getenv("VERCEL") else os.path.join(ROOT_DIR, "predictions.db"))
MODEL_PATH = os.path.join(ROOT_DIR, "covid19Model.pkl")

ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "Vikash09")
SECRET_KEY     = os.environ.get("SECRET_KEY",     "covid19-prediction-secret")

AGE_MEAN = 41.794102472403026
AGE_STD  = 16.907381137350168

YES_NO = {"Yes":1,"No":0,"yes":1,"no":0,True:1,False:0,1:1,0:0}
GENDER = {"Female":0,"Male":1,"female":0,"male":1,0:0,1:1}

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

app.secret_key = SECRET_KEY

# ─────────────────────────────────────────────
# PSYCOPG2 → SQLITE3 COMPATIBILITY LAYER
# Wraps psycopg2 so all existing conn.execute() / cursor.fetchone() / lastrowid
# calls work without any changes elsewhere in the file.
# ─────────────────────────────────────────────

# SQLite DDL → PostgreSQL DDL conversions
_PG_DDL_SUBS = [
    ("INTEGER PRIMARY KEY AUTOINCREMENT", "SERIAL PRIMARY KEY"),
    ("INTEGER  PRIMARY KEY AUTOINCREMENT", "SERIAL PRIMARY KEY"),
]

def _fix_pg_ddl(sql: str) -> str:
    for old, new in _PG_DDL_SUBS:
        sql = sql.replace(old, new)
    return sql


class _CompatRow:
    """
    A database row that supports BOTH dict-style (row['col']) and
    index-style (row[0]) access, so sqlite3.Row and psycopg2 rows
    are interchangeable everywhere in the code.
    """
    __slots__ = ("_data", "_keys", "_dict")

    def __init__(self, data, keys):
        self._data = list(data)
        self._keys = list(keys)
        self._dict = dict(zip(self._keys, self._data))

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._data[key]
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def get(self, key, default=None):
        return self._dict.get(key, default)

    def keys(self):
        return self._keys

    def items(self):
        return self._dict.items()

    def values(self):
        return self._data

    def __iter__(self):
        # Makes dict(row) work: Python calls keys() then row[key]
        return iter(self._data)

    def __repr__(self):
        return repr(self._dict)


class _PgCursor:
    """Wraps a psycopg2 cursor to match the sqlite3 cursor API."""

    def __init__(self, cur, lastrowid=None):
        self._cur = cur
        self.lastrowid = lastrowid

    def _wrap(self, row):
        if row is None:
            return None
        cols = [d[0] for d in self._cur.description]
        return _CompatRow(row, cols)

    def fetchone(self):
        return self._wrap(self._cur.fetchone())

    def fetchall(self):
        if not self._cur.description:
            return []
        cols = [d[0] for d in self._cur.description]
        return [_CompatRow(r, cols) for r in self._cur.fetchall()]


class _PgConnection:
    """
    Makes psycopg2 look like a sqlite3 connection:
      • Converts '?' placeholders → '%s'
      • Auto-appends RETURNING id to INSERT statements → sets cursor.lastrowid
      • executescript() converts SQLite DDL → PostgreSQL DDL, then runs statements
      • Context-manager __exit__ commits on success, rolls back on exception
    """

    def __init__(self, dsn: str):
        self._conn = psycopg2.connect(dsn)
        self._conn.autocommit = False

    # ── internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _to_pg_placeholders(sql: str) -> str:
        return sql.replace("?", "%s")

    def execute(self, sql: str, params=None):
        sql = self._to_pg_placeholders(sql)
        cur = self._conn.cursor()
        lastrowid = None
        is_insert = sql.strip().upper().startswith("INSERT")
        if is_insert and "RETURNING" not in sql.upper():
            sql = sql.rstrip().rstrip(";") + " RETURNING id"
        cur.execute(sql, params or [])
        if is_insert:
            row = cur.fetchone()
            lastrowid = row[0] if row else None
        return _PgCursor(cur, lastrowid)

    def executescript(self, script: str):
        """Run multiple semicolon-separated SQL statements (used by init_db)."""
        cur = self._conn.cursor()
        for stmt in script.split(";"):
            stmt = stmt.strip()
            if stmt:
                stmt = _fix_pg_ddl(stmt)
                cur.execute(stmt)
        self._conn.commit()

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self._conn.rollback()
        else:
            self._conn.commit()
        return False  # never suppress exceptions

# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────
def get_db():
    """Return a database connection — Supabase (PostgreSQL) or SQLite."""
    if USE_POSTGRES:
        return _PgConnection(DATABASE_URL)
    # Ensure the directory for the DB file exists (needed on Vercel /tmp)
    db_dir = os.path.dirname(DB_PATH)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

class _SqliteCtx:
    """Context manager wrapper for sqlite3 connection so 'with get_db()' works correctly."""
    def __init__(self, conn):
        self._conn = conn
    def __getattr__(self, name):
        return getattr(self._conn, name)
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self._conn.commit()
        else:
            self._conn.rollback()
        self._conn.close()
        return False

def get_db_ctx():
    """Return a context-manager-safe DB connection."""
    if USE_POSTGRES:
        return _PgConnection(DATABASE_URL)
    return _SqliteCtx(sqlite3.connect(DB_PATH) if True else None)

def _make_sqlite_ctx():
    db_dir = os.path.dirname(DB_PATH)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return _SqliteCtx(conn)

_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name   TEXT    NOT NULL DEFAULT '',
    username    TEXT    NOT NULL UNIQUE,
    email       TEXT    NOT NULL DEFAULT '',
    password    TEXT    NOT NULL,
    age         INTEGER NOT NULL DEFAULT 25,
    gender      TEXT    NOT NULL DEFAULT 'Male',
    created_at  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id             INTEGER,
    created_at          TEXT    NOT NULL,
    name                TEXT    NOT NULL DEFAULT '',
    phone               TEXT    NOT NULL DEFAULT '',
    email               TEXT    NOT NULL DEFAULT '',
    address             TEXT    NOT NULL DEFAULT '',
    age                 INTEGER NOT NULL DEFAULT 35,
    gender              TEXT    NOT NULL DEFAULT 'Male',
    fever               TEXT    NOT NULL DEFAULT 'No',
    cough               TEXT    NOT NULL DEFAULT 'No',
    cold_sore_throat    TEXT    NOT NULL DEFAULT 'No',
    weakness            TEXT    NOT NULL DEFAULT 'No',
    breathing_difficulty TEXT   NOT NULL DEFAULT 'No',
    high_temperature    TEXT    NOT NULL DEFAULT 'No',
    temperature_f       REAL    NOT NULL DEFAULT 98.6,
    chest_pain          TEXT    NOT NULL DEFAULT 'No',
    loss_smell_taste    TEXT    NOT NULL DEFAULT 'No',
    diabetes            TEXT    NOT NULL DEFAULT 'No',
    asthma              TEXT    NOT NULL DEFAULT 'No',
    smoke               TEXT    NOT NULL DEFAULT 'No',
    high_blood_pressure TEXT    NOT NULL DEFAULT 'No',
    heart_disease       TEXT    NOT NULL DEFAULT 'No',
    risk_label          TEXT    NOT NULL DEFAULT 'Low Risk',
    risk_probability    REAL    NOT NULL DEFAULT 0,
    prediction          INTEGER NOT NULL DEFAULT 0,
    recommendation      TEXT    NOT NULL DEFAULT '',
    fallback            INTEGER NOT NULL DEFAULT 0
);
"""

def init_db():
    """Create tables if they don't exist. Safe to call multiple times."""
    try:
        if USE_POSTGRES:
            with _PgConnection(DATABASE_URL) as conn:
                conn.executescript(_DB_SCHEMA)
        else:
            db_dir = os.path.dirname(DB_PATH)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            conn.executescript(_DB_SCHEMA)
            conn.commit()
            conn.close()
    except Exception as _e:
        # Log but don't crash — app can still serve requests even if DB init
        # fails transiently (e.g., Vercel cold start race condition)
        import sys
        print(f"[init_db] WARNING: {_e}", file=sys.stderr)

init_db()


# ─────────────────────────────────────────────
# AUTH HELPERS
# ─────────────────────────────────────────────
def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def get_user_by_id(uid: int):
    with get_db() as conn:
        return conn.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()

def get_user_by_username(username: str):
    with get_db() as conn:
        return conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()

def is_logged_in() -> bool:
    return bool(session.get("user_id"))

def current_user():
    uid = session.get("user_id")
    return get_user_by_id(uid) if uid else None

def is_admin() -> bool:
    return bool(session.get("is_admin"))

# ─────────────────────────────────────────────
# ML MODEL
# ─────────────────────────────────────────────
@lru_cache(maxsize=1)
def load_model():
    try:
        model = pd.read_pickle(MODEL_PATH)
        if not hasattr(model, "predict"):
            raise TypeError("Not a valid model")
        return model
    except Exception:
        return None

def safe_int(v, default=0, lo=None, hi=None):
    try:
        n = int(float(v))
    except Exception:
        n = default
    if lo is not None: n = max(lo, n)
    if hi is not None: n = min(hi, n)
    return n

def safe_float(v, default=0.0, lo=None, hi=None):
    try:
        n = float(v)
        if not np.isfinite(n): n = default
    except Exception:
        n = default
    if lo is not None: n = max(lo, n)
    if hi is not None: n = min(hi, n)
    return n

def yn(v) -> int:
    return int(YES_NO.get(v, 0))

def symptom_severity(a: dict) -> int:
    keys = ["fever","cough","cold_sore_throat","weakness",
            "breathing_difficulty","high_temperature","chest_pain","loss_smell_taste"]
    return sum(a[k] for k in keys)

def classification_from_symptoms(a: dict, temp_f: float) -> int:
    sev = symptom_severity(a)
    if a["breathing_difficulty"] or a["chest_pain"] or temp_f >= 103: return 3
    if sev >= 5 or temp_f >= 101.5: return 2
    if sev >= 3: return 4
    if sev >= 1: return 5
    return 7

def build_answers(payload: dict) -> dict:
    age   = safe_int(payload.get("age"), 35, 0, 121)
    temp  = safe_float(payload.get("temperature_f"), 98.6, 90.0, 110.0)
    ht    = yn(payload.get("high_temperature"))
    return {
        "age": age,
        "gender": int(GENDER.get(payload.get("gender","Male"), 1)),
        "fever": yn(payload.get("fever")),
        "cough": yn(payload.get("cough")),
        "cold_sore_throat": yn(payload.get("cold_sore_throat")),
        "weakness": yn(payload.get("weakness")),
        "breathing_difficulty": yn(payload.get("breathing_difficulty")),
        "high_temperature": 1 if ht or temp >= 100.4 else 0,
        "temperature_f": temp,
        "chest_pain": yn(payload.get("chest_pain")),
        "diabetes": yn(payload.get("diabetes")),
        "asthma": yn(payload.get("asthma")),
        "smoke": yn(payload.get("smoke")),
        "high_blood_pressure": yn(payload.get("high_blood_pressure")),
        "heart_disease": yn(payload.get("heart_disease")),
        "loss_smell_taste": yn(payload.get("loss_smell_taste")),
    }

def build_model_input(model, answers: dict) -> pd.DataFrame:
    if not hasattr(model, "feature_names_in_"):
        raise ValueError("Model missing feature_names_in_")
    features = list(model.feature_names_in_)
    row = {f: 0.0 for f in features}
    sev = symptom_severity(answers)
    cls = classification_from_symptoms(answers, answers["temperature_f"])
    row.update({
        "USMER": 1.0,
        "PATIENT_TYPE": 0.0 if answers["breathing_difficulty"] or answers["chest_pain"] or sev >= 6 else 1.0,
        "INTUBED": 1.0 if answers["breathing_difficulty"] and answers["chest_pain"] else 0.0,
        "PNEUMONIA": 1.0 if answers["breathing_difficulty"] or answers["cough"] else 0.0,
        "PREGNANT": 0.0,
        "DIABETES": float(answers["diabetes"]),
        "COPD": 0.0,
        "ASTHMA": float(answers["asthma"]),
        "INMSUPR": 0.0,
        "HIPERTENSION": float(answers["high_blood_pressure"]),
        "OTHER_DISEASE": 1.0 if any(answers[k] for k in ["diabetes","asthma","heart_disease","high_blood_pressure"]) else 0.0,
        "CARDIOVASCULAR": float(answers["heart_disease"]),
        "OBESITY": 0.0,
        "RENAL_CHRONIC": 0.0,
        "TOBACCO": float(answers["smoke"]),
        "ICU": 1.0 if answers["breathing_difficulty"] and answers["chest_pain"] else 0.0,
        "SEX_2": float(answers["gender"]),
        "AGE_scaled": float((answers["age"] - AGE_MEAN) / AGE_STD),
    })
    cf = f"CLASIFFICATION_FINAL_{cls}"
    if cf in row: row[cf] = 1.0
    if "MEDICAL_UNIT_4" in row: row["MEDICAL_UNIT_4"] = 1.0
    return pd.DataFrame([[row[f] for f in features]], columns=features).astype(np.float64)

def risk_details(prob: float):
    if prob < 0.35:
        return "Low Risk", "Your current profile appears lower risk. Continue monitoring symptoms, rest, hydrate, and follow local public-health guidance."
    if prob < 0.65:
        return "Medium Risk", "Your answers suggest meaningful risk. Consider testing, reduce contact with others, and speak with a clinician if symptoms persist or worsen."
    return "High Risk", "Your answers indicate elevated clinical risk. Seek medical guidance promptly, especially if breathing difficulty, chest pain, confusion, or persistent high fever is present."

def run_prediction(model, answers: dict):
    model_input = build_model_input(model, answers)
    pred = int(model.predict(model_input)[0])
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(model_input)[0]
        c2i   = {int(lbl): i for i, lbl in enumerate(model.classes_)}
        ri    = c2i.get(1)
        pi    = c2i.get(pred)
        rp    = float(probs[ri]) if ri is not None else float(pred == 1)
        conf  = float(probs[pi]) if pi is not None else float(np.max(probs))
    else:
        rp, conf = float(pred), 1.0
    return pred, max(0.0, min(rp, 1.0)), max(0.0, min(conf, 1.0))

def fallback_prediction(answers: dict):
    sev  = symptom_severity(answers)
    hr   = sum(answers[k] for k in ["diabetes","asthma","smoke","high_blood_pressure","heart_disease"])
    emg  = answers["breathing_difficulty"] + answers["chest_pain"]
    age_r = 1 if answers["age"] >= 60 else 0
    score = sev*0.09 + hr*0.08 + emg*0.18 + age_r*0.12
    if answers["temperature_f"] >= 103: score += 0.2
    elif answers["temperature_f"] >= 101: score += 0.1
    rp = max(0.05, min(score, 0.95))
    return (1 if rp >= 0.5 else 0), rp, 0.75

def save_prediction(payload: dict, user_id, risk_label, risk_prob, pred, rec, used_fallback) -> int:
    fever = "Yes" if yn(payload.get("fever")) else "No"
    temp  = safe_float(payload.get("temperature_f"), 101.0 if fever=="Yes" else 98.6, 90.0, 110.0)
    def ryn(k): return "Yes" if yn(payload.get(k)) else "No"
    rec_data = {
        "user_id": user_id,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "name": str(payload.get("name","Unknown")).strip() or "Unknown",
        "phone": str(payload.get("phone","")).strip(),
        "email": str(payload.get("email","")).strip(),
        "address": str(payload.get("address","")).strip(),
        "age": safe_int(payload.get("age"), 35, 0, 121),
        "gender": str(payload.get("gender","Male")).strip(),
        "fever": fever,
        "cough": ryn("cough"),
        "cold_sore_throat": ryn("cold_sore_throat"),
        "weakness": ryn("weakness"),
        "breathing_difficulty": ryn("breathing_difficulty"),
        "high_temperature": ryn("high_temperature"),
        "temperature_f": temp,
        "chest_pain": ryn("chest_pain"),
        "loss_smell_taste": ryn("loss_smell_taste"),
        "diabetes": ryn("diabetes"),
        "asthma": ryn("asthma"),
        "smoke": ryn("smoke"),
        "high_blood_pressure": ryn("high_blood_pressure"),
        "heart_disease": ryn("heart_disease"),
        "risk_label": risk_label,
        "risk_probability": round(risk_prob * 100, 2),
        "prediction": pred,
        "recommendation": rec,
        "fallback": int(used_fallback),
    }
    cols = ", ".join(rec_data.keys())
    ph   = ", ".join(["?"] * len(rec_data))
    with get_db() as conn:
        cur = conn.execute(f"INSERT INTO predictions ({cols}) VALUES ({ph})", list(rec_data.values()))
        return int(cur.lastrowid)

def prediction_rows(user_id=None):
    init_db()
    with get_db() as conn:
        if user_id:
            return conn.execute("SELECT * FROM predictions WHERE user_id=? ORDER BY id DESC", (user_id,)).fetchall()
        return conn.execute("SELECT * FROM predictions ORDER BY id DESC").fetchall()

def prediction_row(record_id: int):
    init_db()
    with get_db() as conn:
        return conn.execute("SELECT * FROM predictions WHERE id=?", (record_id,)).fetchone()

def report_context(row):
    return {
        "record": row,
        "symptoms": [
            ("Fever", row["fever"]),
            ("Cough", row["cough"]),
            ("Cold / Sore Throat", row["cold_sore_throat"]),
            ("Fatigue / Weakness", row["weakness"]),
            ("Breathing Difficulty", row["breathing_difficulty"]),
            ("High Temperature", row["high_temperature"]),
            ("Chest Pain", row["chest_pain"]),
            ("Loss of Smell/Taste", row["loss_smell_taste"]),
        ],
        "history": [
            ("Diabetes", row["diabetes"]),
            ("Asthma", row["asthma"]),
            ("Smoking", row["smoke"]),
            ("High Blood Pressure", row["high_blood_pressure"]),
            ("Heart Disease", row["heart_disease"]),
        ],
    }

# ─────────────────────────────────────────────
# ROUTES — PUBLIC
# ─────────────────────────────────────────────
@app.get("/")
@app.get("/index.html")
def home():
    user = current_user()
    init_db()
    with get_db() as conn:
        try:
            # active_users represents how much they use the prediction system (total predictions count)
            active_users = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        except Exception:
            active_users = 0
        
        # lives_impacted is hardcoded to 7,000,000 (7 million)
        lives_impacted = 7000000
        
        # global_cases is 770,000,000 (770 million)
        global_cases = 770000000
        
    return render_template("index.html", user=user, logged_in=is_logged_in(),
                           global_cases=global_cases,
                           active_users=active_users, lives_impacted=lives_impacted,
                           model_accuracy=98.4)

# ── User register ──
@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        # get_json(silent=True) returns None when Content-Type is not JSON;
        # an *actual* empty JSON body returns {} (dict) which is abuse-safe here.
        _raw  = request.get_json(silent=True)
        data  = _raw if isinstance(_raw, dict) else request.form
        full_name = str(data.get("full_name","")).strip()
        username  = str(data.get("username","")).strip()
        email     = str(data.get("email","")).strip()
        password  = str(data.get("password","")).strip()
        age       = safe_int(data.get("age"), 25, 1, 120)
        gender    = str(data.get("gender","Male")).strip()

        if not username or not password:
            if request.is_json:
                return jsonify({"error": "Username and password are required."}), 400
            return redirect(url_for("login_page") + "?error=Username+and+password+required")

        existing = get_user_by_username(username)
        if existing:
            if request.is_json:
                return jsonify({"error": "Username already taken."}), 409
            return render_template(
                "register.html",
                error="Username already taken.",
                full_name=full_name,
                username=username,
                email=email,
                age=age,
                gender=gender
            )

        with get_db() as conn:
            conn.execute(
                "INSERT INTO users (full_name,username,email,password,age,gender,created_at) VALUES (?,?,?,?,?,?,?)",
                (full_name, username, email, hash_pw(password), age, gender, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
        if request.is_json:
            return jsonify({"success": True, "message": "Account created successfully."})
        return redirect(url_for("login_page") + "?success=Account+created")
    return render_template("register.html", error=request.args.get("error",""))

# ── User login ──
@app.route("/login", methods=["GET","POST"])
def login_page():
    if request.method == "POST":
        data     = request.get_json(silent=True) or request.form
        username = str(data.get("username","")).strip()
        password = str(data.get("password","")).strip()
        user = get_user_by_username(username)
        if user and user["password"] == hash_pw(password):
            session["user_id"]  = user["id"]
            session["username"] = user["username"]
            session["is_admin"] = False
            if request.is_json:
                return jsonify({"success": True, "username": user["username"], "full_name": user["full_name"]})
            return redirect(url_for("home"))
        if request.is_json:
            return jsonify({"error": "Invalid username or password."}), 401
        return redirect(url_for("login_page") + "?error=Invalid+credentials")
    return render_template("login.html",
                           error=request.args.get("error",""),
                           success=request.args.get("success",""))

# ── User logout ──
@app.get("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

# ── User dashboard (past predictions + improvement) ──

def _fig_to_b64(fig):
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    plt, _ = _get_matplotlib()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=110,
                facecolor='#0B0F10', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode('utf-8')

def _dark_style(ax):
    """Apply the dark theme to a matplotlib axis."""
    ax.set_facecolor('#0B0F10')
    ax.tick_params(colors='#94a3b8', labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e2a2e')
    ax.xaxis.label.set_color('#94a3b8')
    ax.yaxis.label.set_color('#94a3b8')
    ax.title.set_color('#00FFC6')

def generate_dashboard_graphs(records):
    """Generate matplotlib charts for customer dashboard. Returns dict of b64 strings."""
    plt, mpatches = _get_matplotlib()
    graphs = {}
    if not records:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.set_facecolor('#0B0F10')
        fig.patch.set_facecolor('#0B0F10')
        ax.text(0.5, 0.5, 'No prediction data yet.\nComplete a risk assessment\nto see your graphs here.',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, color='#94a3b8', fontstyle='italic')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        graphs['risk_trend'] = _fig_to_b64(fig)
        return graphs

    recs = records[::-1]  # oldest → newest

    # ── Graph 1: Risk Probability Trend ──
    labels = [r.get('created_at','')[5:16] for r in recs]  # MM-DD HH:MM
    probs  = [float(r.get('risk_probability') or 0) for r in recs]
    fig1, ax1 = plt.subplots(figsize=(7, 2.8))
    _dark_style(ax1)
    colors = ['#ef4444' if p >= 65 else '#f59e0b' if p >= 35 else '#10b981' for p in probs]
    bars = ax1.bar(range(len(labels)), probs, color=colors, alpha=0.85, width=0.55, zorder=3)
    ax1.plot(range(len(labels)), probs, color='#00FFC6', linewidth=1.5, marker='o',
             markersize=5, markerfacecolor='#00FFC6', zorder=4)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=30, ha='right', fontsize=7.5)
    ax1.set_ylim(0, 110)
    ax1.set_ylabel('Risk %', fontsize=9)
    ax1.set_title('Risk Probability Over Time', fontsize=11, fontweight='bold', pad=8)
    ax1.yaxis.grid(True, color='#1e2a2e', zorder=0)
    ax1.set_axisbelow(True)
    green_patch  = mpatches.Patch(color='#10b981', label='Low Risk (<35%)')
    yellow_patch = mpatches.Patch(color='#f59e0b', label='Medium Risk (35-65%)')
    red_patch    = mpatches.Patch(color='#ef4444', label='High Risk (>65%)')
    ax1.legend(handles=[green_patch, yellow_patch, red_patch],
               fontsize=7.5, facecolor='#0B0F10', edgecolor='#1e2a2e',
               labelcolor='#94a3b8', loc='upper left')
    graphs['risk_trend'] = _fig_to_b64(fig1)

    # ── Graph 2: Symptoms Frequency ──
    SYMPTOM_COLS = ['fever','cough','cold_sore_throat','weakness',
                    'breathing_difficulty','high_temperature','chest_pain','loss_smell_taste']
    sym_counts = {s: sum(1 for r in recs if str(r.get(s,'')).lower() in ('yes','1','true')) for s in SYMPTOM_COLS}
    sym_labels = {'fever':'Fever','cough':'Cough','cold_sore_throat':'Cold/Sore Throat',
                  'weakness':'Weakness','breathing_difficulty':'Breathing Diff.',
                  'high_temperature':'High Temp','chest_pain':'Chest Pain','loss_smell_taste':'Smell/Taste Loss'}
    sym_names  = [sym_labels.get(s, s) for s in SYMPTOM_COLS]
    sym_vals   = [sym_counts.get(s, 0) for s in SYMPTOM_COLS]
    fig2, ax2  = plt.subplots(figsize=(7, 2.8))
    _dark_style(ax2)
    bar_colors = ['#ef4444' if v > 0 else '#1e5568' for v in sym_vals]
    bars2 = ax2.barh(sym_names, sym_vals, color=bar_colors, alpha=0.85, height=0.6, zorder=3)
    for bar, val in zip(bars2, sym_vals):
        if val > 0:
            ax2.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                     str(val), va='center', fontsize=8.5, color='#F8FAFC')
    ax2.set_xlabel('Times Reported', fontsize=9)
    ax2.set_title('Symptom Occurrence History', fontsize=11, fontweight='bold', pad=8)
    ax2.set_xlim(0, max(sym_vals) + 1.2 if sym_vals else 1)
    ax2.xaxis.grid(True, color='#1e2a2e', zorder=0)
    ax2.set_axisbelow(True)
    ax2.spines['left'].set_visible(False)
    graphs['symptoms'] = _fig_to_b64(fig2)

    # ── Graph 3: Temperature History ──
    temps  = [float(r.get('temperature_f') or 98.6) for r in recs]
    dates  = [r.get('created_at','')[5:16] for r in recs]
    fig3, ax3 = plt.subplots(figsize=(7, 2.8))
    _dark_style(ax3)
    ax3.fill_between(range(len(temps)), temps, alpha=0.12, color='#06B6D4')
    ax3.plot(range(len(temps)), temps, color='#06B6D4', linewidth=2, marker='s',
             markersize=5, markerfacecolor='#06B6D4', zorder=3)
    for i, t in enumerate(temps):
        color = '#ef4444' if t >= 102 else '#f59e0b' if t >= 100.4 else '#10b981'
        ax3.annotate(f'{t:.1f}°F', (i, t), textcoords='offset points',
                     xytext=(0, 8), ha='center', fontsize=7.5, color=color, fontweight='bold')
    ax3.set_xticks(range(len(dates)))
    ax3.set_xticklabels(dates, rotation=30, ha='right', fontsize=7.5)
    ax3.set_ylabel('Temperature °F', fontsize=9)
    ax3.set_title('Body Temperature Trend', fontsize=11, fontweight='bold', pad=8)
    ax3.yaxis.grid(True, color='#1e2a2e', zorder=0)
    ax3.axhline(y=100.4, color='#f59e0b', linestyle='--', linewidth=1, alpha=0.7, label='Fever threshold')
    ax3.axhline(y=102.0, color='#ef4444', linestyle='--', linewidth=1, alpha=0.7, label='High fever')
    ax3.legend(fontsize=7.5, facecolor='#0B0F10', edgecolor='#1e2a2e', labelcolor='#94a3b8')
    ax3.set_axisbelow(True)
    graphs['temperature'] = _fig_to_b64(fig3)

    # ── Graph 4: Medical History flags ──
    MED_COLS = ['diabetes','asthma','smoke','high_blood_pressure','heart_disease']
    med_labels = {'diabetes':'Diabetes','asthma':'Asthma','smoke':'Smoking',
                  'high_blood_pressure':'High BP','heart_disease':'Heart Disease'}
    med_vals = [sum(1 for r in recs if str(r.get(s,'')).lower() in ('yes','1','true')) for s in MED_COLS]
    med_names = [med_labels.get(s, s) for s in MED_COLS]
    fig4, ax4   = plt.subplots(figsize=(7, 2.8))
    _dark_style(ax4)
    med_bars = ax4.bar(med_names, med_vals, color='#7C3AED', alpha=0.8, width=0.5, zorder=3)
    for bar, val in zip(med_bars, med_vals):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 str(val), ha='center', va='bottom', fontsize=9, color='#F8FAFC', fontweight='bold')
    ax4.set_ylabel('Times Reported', fontsize=9)
    ax4.set_title('Medical History Occurrence', fontsize=11, fontweight='bold', pad=8)
    ax4.set_ylim(0, max(med_vals) + 1.2 if med_vals else 1)
    ax4.yaxis.grid(True, color='#1e2a2e', zorder=0)
    ax4.set_axisbelow(True)
    graphs['medical_history'] = _fig_to_b64(fig4)

    return graphs

@app.get("/dashboard")
def user_dashboard():
    if not is_logged_in():
        return redirect(url_for("login_page"))
    user  = current_user()
    rows  = prediction_rows(user_id=user["id"])
    records = [dict(r) for r in rows]
    # Compute improvement: compare latest vs previous risk_probability
    improvement = None
    if len(records) >= 2:
        latest = records[0]["risk_probability"]
        prev   = records[1]["risk_probability"]
        improvement = round(prev - latest, 2)   # positive = improved
    # Generate matplotlib graphs
    graphs = generate_dashboard_graphs(records)
    return render_template("customer_dashboard.html",
                           user=user,
                           records=records,
                           improvement=improvement,
                           logged_in=True,
                           graphs=graphs)

# ─────────────────────────────────────────────
# ROUTES — PREDICT
# ─────────────────────────────────────────────
@app.after_request
def cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

@app.route("/predict", methods=["OPTIONS"])
def predict_options():
    return "", 204

@app.post("/predict")
def predict_risk():
    try:
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            payload = {}

        # DEBUG: keep server from crashing on bad payload; return explicit error
        # (helps during local testing)

        answers      = build_answers(payload)
        used_fallback = False
        try:
            model = load_model()
            if model is None: raise RuntimeError("Model not loaded")
            model_input = build_model_input(model, answers)
            pred, rp, conf = run_prediction(model, answers)
        except Exception:
            pred, rp, conf = fallback_prediction(answers)
            used_fallback = True

        risk_label, rec = risk_details(rp)
        user_id = session.get("user_id")   # None if not logged in — still saves

        # If we cannot save the record (e.g., user_id None, db issue), still return prediction.
        record_id = None
        try:
            if user_id is None:
                # allow anonymous saves by storing user_id as NULL (sqlite supports it)
                record_id = save_prediction(payload, None, risk_label, rp, pred, rec, used_fallback)
            else:
                record_id = save_prediction(payload, user_id, risk_label, rp, pred, rec, used_fallback)
        except Exception:
            record_id = None

        # ensure payload is always JSON serializable (prevents server crash/reset)
        return jsonify({
            "record_id":       record_id,
            "risk_label":      str(risk_label),
            "risk_probability": float(rp),
            "confidence":      float(conf),
            "prediction":      int(pred),
            "recommendation":  str(rec),
            "fallback":        int(used_fallback),
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

# ─────────────────────────────────────────────
# ROUTES — REPORTS & DOWNLOADS
# ─────────────────────────────────────────────
@app.get("/report/<int:record_id>")
def report(record_id):
    row = prediction_row(record_id)
    if row is None:
        return jsonify({"error": "Record not found."}), 404
    return render_template("report.html", **report_context(row))

@app.get("/report.html")
def latest_report():
    rows = prediction_rows()
    if not rows: return redirect(url_for("home"))
    return redirect(url_for("report", record_id=rows[0]["id"]))

@app.get("/download/<int:record_id>")
def download_prediction(record_id):
    row = prediction_row(record_id)
    if row is None: return jsonify({"error": "Not found."}), 404
    out = io.StringIO()
    w   = csv.DictWriter(out, fieldnames=row.keys())
    w.writeheader(); w.writerow(dict(row))
    return Response(out.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=prediction_{record_id}.csv"})

@app.get("/download/<int:record_id>/word")
def download_word(record_id):
    row = prediction_row(record_id)
    if row is None: return jsonify({"error": "Not found."}), 404
    html = render_template("report.html", printable=False, **report_context(row))
    return Response(html, mimetype="application/msword",
                    headers={"Content-Disposition": f"attachment; filename=covid_report_{record_id}.doc"})

# ─────────────────────────────────────────────
# ROUTES — ADMIN
# ─────────────────────────────────────────────
@app.route("/admin", methods=["GET"])
@app.route("/admin.html", methods=["GET"])
def admin():
    if not is_admin():
        return render_template("admin.html", login_mode=True,
                               records=None, error=request.args.get("error",""))
    records = prediction_rows()
    return render_template("admin.html", login_mode=False,
                           records=records, error="")

@app.route("/admin/login", methods=["POST"])
@app.route("/admin_login.html", methods=["POST"])
def admin_login():
    pw = request.form.get("password","")
    if pw == ADMIN_PASSWORD:
        session["is_admin"] = True
        return redirect(url_for("admin"))
    return redirect(url_for("admin") + "?error=Invalid+admin+password")

@app.get("/admin/logout")
def admin_logout():
    session.pop("is_admin", None)
    return redirect(url_for("admin"))

@app.get("/admin/download")
def admin_download():
    if not is_admin(): return redirect(url_for("admin"))
    rows = prediction_rows()
    out  = io.StringIO()
    if rows:
        w = csv.DictWriter(out, fieldnames=rows[0].keys())
        w.writeheader()
        for r in rows: w.writerow(dict(r))
    else:
        out.write("No records\n")
    return Response(out.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=all_predictions.csv"})

# Backward-compatible alias: some UI files may reference /user_login.html
@app.get("/user_login.html")
def user_login_alias():
    return redirect(url_for("login_page"))

# Register HTML-only route (requested working URL)
@app.get("/register.html")
def register_html_only():
    # Render the register page without interfering with the existing /register POST handler
    return render_template("register.html", error="")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
