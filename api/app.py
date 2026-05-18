from functools import lru_cache
import csv
import os
import io
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, redirect, render_template, request, session, url_for


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

# For Vercel serverless functions, use /tmp for writable files like SQLite database
if os.getenv("VERCEL"):
    DB_PATH = "/tmp/predictions.db"
else:
    DB_PATH = os.path.join(ROOT_DIR, "predictions.db")

MODEL_PATH = os.path.join(ROOT_DIR, "covid19Model.pkl")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "Vikash09")

AGE_MEAN = 41.794102472403026
AGE_STD = 16.907381137350168

YES_NO = {"Yes": 1, "No": 0, "yes": 1, "no": 0, True: 1, False: 0, 1: 1, 0: 0}
GENDER = {"Female": 0, "Male": 1, "female": 0, "male": 1, 0: 0, 1: 1}

PREDICTION_COLUMNS = {
    "phone": "TEXT NOT NULL DEFAULT ''",
    "email": "TEXT NOT NULL DEFAULT ''",
    "address": "TEXT NOT NULL DEFAULT ''",
    "age": "INTEGER NOT NULL DEFAULT 35",
    "gender": "TEXT NOT NULL DEFAULT 'Male'",
    "fever": "TEXT NOT NULL DEFAULT 'No'",
    "cough": "TEXT NOT NULL DEFAULT 'No'",
    "cold_sore_throat": "TEXT NOT NULL DEFAULT 'No'",
    "weakness": "TEXT NOT NULL DEFAULT 'No'",
    "breathing_difficulty": "TEXT NOT NULL DEFAULT 'No'",
    "high_temperature": "TEXT NOT NULL DEFAULT 'No'",
    "temperature_f": "REAL NOT NULL DEFAULT 98.6",
    "chest_pain": "TEXT NOT NULL DEFAULT 'No'",
    "loss_smell_taste": "TEXT NOT NULL DEFAULT 'No'",
    "diabetes": "TEXT NOT NULL DEFAULT 'No'",
    "asthma": "TEXT NOT NULL DEFAULT 'No'",
    "smoke": "TEXT NOT NULL DEFAULT 'No'",
    "high_blood_pressure": "TEXT NOT NULL DEFAULT 'No'",
    "heart_disease": "TEXT NOT NULL DEFAULT 'No'",
    "risk_label": "TEXT NOT NULL DEFAULT 'Low Risk'",
    "risk_probability": "REAL NOT NULL DEFAULT 0",
    "prediction": "INTEGER NOT NULL DEFAULT 0",
    "recommendation": "TEXT NOT NULL DEFAULT ''",
}

app = Flask(__name__, template_folder=ROOT_DIR)
app.secret_key = os.environ.get("SECRET_KEY", "covid-risk-local-secret")


def get_db():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    with get_db() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                name TEXT NOT NULL,
                phone TEXT NOT NULL DEFAULT '',
                email TEXT NOT NULL DEFAULT '',
                address TEXT NOT NULL DEFAULT '',
                age INTEGER NOT NULL,
                gender TEXT NOT NULL,
                fever TEXT NOT NULL,
                cough TEXT NOT NULL,
                cold_sore_throat TEXT NOT NULL,
                weakness TEXT NOT NULL,
                breathing_difficulty TEXT NOT NULL,
                high_temperature TEXT NOT NULL,
                temperature_f REAL NOT NULL,
                chest_pain TEXT NOT NULL,
                loss_smell_taste TEXT NOT NULL,
                diabetes TEXT NOT NULL,
                asthma TEXT NOT NULL,
                smoke TEXT NOT NULL,
                high_blood_pressure TEXT NOT NULL,
                heart_disease TEXT NOT NULL,
                risk_label TEXT NOT NULL,
                risk_probability REAL NOT NULL,
                prediction INTEGER NOT NULL,
                recommendation TEXT NOT NULL
            )
            """
        )
        existing_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(predictions)").fetchall()
        }

        for column, definition in PREDICTION_COLUMNS.items():
            if column not in existing_columns:
                connection.execute(f"ALTER TABLE predictions ADD COLUMN {column} {definition}")


def clean_text(value, default=""):
    text = str(value or default).strip()
    return text if text else default


def safe_int(value, default=0, minimum=None, maximum=None) -> int:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        number = default

    if minimum is not None:
        number = max(minimum, number)
    if maximum is not None:
        number = min(maximum, number)

    return number


def safe_float(value, default=0.0, minimum=None, maximum=None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default

    if not np.isfinite(number):
        number = default
    if minimum is not None:
        number = max(minimum, number)
    if maximum is not None:
        number = min(maximum, number)

    return number


def raw_yes_no(payload: dict, key: str) -> str:
    return "Yes" if to_yes_no(payload.get(key)) else "No"


@lru_cache(maxsize=1)
def load_model():
    try:
        model = pd.read_pickle(MODEL_PATH)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}") from exc

    if not hasattr(model, "predict"):
        raise TypeError("Loaded object does not have a predict() method.")

    return model


def to_yes_no(value) -> int:
    return int(YES_NO.get(value, 0))


def to_gender(value) -> int:
    return int(GENDER.get(value, 1))


def symptom_severity(values: dict) -> int:
    symptom_keys = [
        "fever",
        "cough",
        "cold_sore_throat",
        "weakness",
        "breathing_difficulty",
        "high_temperature",
        "chest_pain",
        "loss_smell_taste",
    ]
    return int(sum(values[key] for key in symptom_keys))


def classification_from_symptoms(values: dict, temperature_f: float) -> int:
    severity = symptom_severity(values)

    if values["breathing_difficulty"] or values["chest_pain"] or temperature_f >= 103:
        return 3
    if severity >= 5 or temperature_f >= 101.5:
        return 2
    if severity >= 3:
        return 4
    if severity >= 1:
        return 5
    return 7


def build_answers(payload: dict) -> dict:
    age = safe_int(payload.get("age"), default=35, minimum=0, maximum=121)
    temperature_f = safe_float(payload.get("temperature_f"), default=98.6, minimum=90.0, maximum=110.0)
    high_temperature = to_yes_no(payload.get("high_temperature"))

    return {
        "age": age,
        "gender": to_gender(payload.get("gender", "Male")),
        "fever": to_yes_no(payload.get("fever")),
        "cough": to_yes_no(payload.get("cough")),
        "cold_sore_throat": to_yes_no(payload.get("cold_sore_throat")),
        "weakness": to_yes_no(payload.get("weakness")),
        "breathing_difficulty": to_yes_no(payload.get("breathing_difficulty")),
        "high_temperature": 1 if high_temperature or temperature_f >= 100.4 else 0,
        "temperature_f": temperature_f,
        "chest_pain": to_yes_no(payload.get("chest_pain")),
        "diabetes": to_yes_no(payload.get("diabetes")),
        "asthma": to_yes_no(payload.get("asthma")),
        "smoke": to_yes_no(payload.get("smoke")),
        "high_blood_pressure": to_yes_no(payload.get("high_blood_pressure")),
        "heart_disease": to_yes_no(payload.get("heart_disease")),
        "loss_smell_taste": to_yes_no(payload.get("loss_smell_taste")),
    }


def build_model_input(model, answers: dict) -> pd.DataFrame:
    if not hasattr(model, "feature_names_in_"):
        raise ValueError("The model does not expose feature_names_in_; cannot guarantee feature order.")

    feature_names = list(model.feature_names_in_)
    row = {feature: 0.0 for feature in feature_names}

    severity = symptom_severity(answers)
    classification = classification_from_symptoms(answers, answers["temperature_f"])

    row.update(
        {
            "USMER": 1.0,
            "PATIENT_TYPE": 0.0
            if answers["breathing_difficulty"] or answers["chest_pain"] or severity >= 6
            else 1.0,
            "INTUBED": 1.0 if answers["breathing_difficulty"] and answers["chest_pain"] else 0.0,
            "PNEUMONIA": 1.0 if answers["breathing_difficulty"] or answers["cough"] else 0.0,
            "PREGNANT": 0.0,
            "DIABETES": float(answers["diabetes"]),
            "COPD": 0.0,
            "ASTHMA": float(answers["asthma"]),
            "INMSUPR": 0.0,
            "HIPERTENSION": float(answers["high_blood_pressure"]),
            "OTHER_DISEASE": 1.0
            if answers["diabetes"]
            or answers["asthma"]
            or answers["heart_disease"]
            or answers["high_blood_pressure"]
            else 0.0,
            "CARDIOVASCULAR": float(answers["heart_disease"]),
            "OBESITY": 0.0,
            "RENAL_CHRONIC": 0.0,
            "TOBACCO": float(answers["smoke"]),
            "ICU": 1.0 if answers["breathing_difficulty"] and answers["chest_pain"] else 0.0,
            "SEX_2": float(answers["gender"]),
            "AGE_scaled": float((answers["age"] - AGE_MEAN) / AGE_STD),
        }
    )

    classification_feature = f"CLASIFFICATION_FINAL_{classification}"
    if classification_feature in row:
        row[classification_feature] = 1.0

    if "MEDICAL_UNIT_4" in row:
        row["MEDICAL_UNIT_4"] = 1.0

    model_input = pd.DataFrame([[row[name] for name in feature_names]], columns=feature_names)
    return model_input.astype(np.float64)


def risk_details(probability: float) -> tuple[str, str]:
    if probability < 0.35:
        return (
            "Low Risk",
            "Your current profile appears lower risk. Continue monitoring symptoms, rest, hydrate, and follow local public-health guidance.",
        )
    if probability < 0.65:
        return (
            "Medium Risk",
            "Your answers suggest meaningful risk. Consider testing, reduce contact with others, and speak with a clinician if symptoms persist or worsen.",
        )
    return (
        "High Risk",
        "Your answers indicate elevated clinical risk. Seek medical guidance promptly, especially if breathing difficulty, chest pain, confusion, or persistent high fever is present.",
    )


def predict(model, model_input: pd.DataFrame) -> tuple[int, float, float]:
    prediction = int(model.predict(model_input)[0])

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(model_input)[0]
        class_to_index = {int(label): index for index, label in enumerate(model.classes_)}
        risk_index = class_to_index.get(1)
        prediction_index = class_to_index.get(prediction)
        risk_probability = float(probabilities[risk_index]) if risk_index is not None else float(prediction == 1)
        confidence = float(probabilities[prediction_index]) if prediction_index is not None else float(np.max(probabilities))
    else:
        risk_probability = float(prediction)
        confidence = 1.0

    return prediction, max(0.0, min(risk_probability, 1.0)), max(0.0, min(confidence, 1.0))


def fallback_prediction(answers: dict) -> tuple[int, float, float]:
    severity = symptom_severity(answers)
    health_risks = sum(
        answers[key]
        for key in ("diabetes", "asthma", "smoke", "high_blood_pressure", "heart_disease")
    )
    emergency_symptoms = answers["breathing_difficulty"] + answers["chest_pain"]
    age_risk = 1 if answers["age"] >= 60 else 0

    score = (severity * 0.09) + (health_risks * 0.08) + (emergency_symptoms * 0.18) + (age_risk * 0.12)
    if answers["temperature_f"] >= 103:
        score += 0.2
    elif answers["temperature_f"] >= 101:
        score += 0.1

    risk_probability = max(0.05, min(score, 0.95))
    prediction = 1 if risk_probability >= 0.5 else 0
    return prediction, risk_probability, 0.75


def save_prediction(payload: dict, risk_label: str, risk_probability: float, prediction: int, recommendation: str) -> int:
    init_db()
    fever = raw_yes_no(payload, "fever")
    temperature_f = safe_float(payload.get("temperature_f"), default=101.0 if fever == "Yes" else 98.6, minimum=90.0, maximum=110.0)

    record = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "name": clean_text(payload.get("name"), "Unknown"),
        "phone": clean_text(payload.get("phone"), ""),
        "email": clean_text(payload.get("email"), ""),
        "address": clean_text(payload.get("address"), ""),
        "age": safe_int(payload.get("age"), default=35, minimum=0, maximum=121),
        "gender": clean_text(payload.get("gender"), "Male"),
        "fever": fever,
        "cough": raw_yes_no(payload, "cough"),
        "cold_sore_throat": raw_yes_no(payload, "cold_sore_throat"),
        "weakness": raw_yes_no(payload, "weakness"),
        "breathing_difficulty": raw_yes_no(payload, "breathing_difficulty"),
        "high_temperature": raw_yes_no(payload, "high_temperature"),
        "temperature_f": temperature_f,
        "chest_pain": raw_yes_no(payload, "chest_pain"),
        "loss_smell_taste": raw_yes_no(payload, "loss_smell_taste"),
        "diabetes": raw_yes_no(payload, "diabetes"),
        "asthma": raw_yes_no(payload, "asthma"),
        "smoke": raw_yes_no(payload, "smoke"),
        "high_blood_pressure": raw_yes_no(payload, "high_blood_pressure"),
        "heart_disease": raw_yes_no(payload, "heart_disease"),
        "risk_label": risk_label,
        "risk_probability": round(risk_probability * 100, 2),
        "prediction": prediction,
        "recommendation": recommendation,
    }

    columns = ", ".join(record.keys())
    placeholders = ", ".join(["?"] * len(record))

    with get_db() as connection:
        cursor = connection.execute(
            f"INSERT INTO predictions ({columns}) VALUES ({placeholders})",
            list(record.values()),
        )
        return int(cursor.lastrowid)


def prediction_rows():
    init_db()
    with get_db() as connection:
        return connection.execute("SELECT * FROM predictions ORDER BY id DESC").fetchall()


def prediction_row(record_id: int):
    init_db()
    with get_db() as connection:
        return connection.execute("SELECT * FROM predictions WHERE id = ?", (record_id,)).fetchone()


def rows_to_csv(rows):
    output = io.StringIO()

    if not rows:
        output.write("No records found\n")
        return output.getvalue()

    fieldnames = rows[0].keys()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(dict(row))

    return output.getvalue()


def report_context(row):
    return {
        "record": row,
        "symptoms": [
            ("Fever", row["fever"]),
            ("Cough", row["cough"]),
            ("Cold / Sore Throat", row["cold_sore_throat"]),
            ("Fatigue", row["weakness"]),
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


def is_admin_logged_in():
    return session.get("admin_logged_in") is True


def require_admin():
    if not is_admin_logged_in():
        return redirect(url_for("admin"))

    return None


@app.get("/")
@app.get("/index.html")
def home():
    return render_template("index.html")


@app.route("/admin_login.html", methods=["GET", "POST"])
@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        password = request.form.get("password", "")

        if password == ADMIN_PASSWORD:
            session["admin_logged_in"] = True
            return redirect(url_for("admin"))

        return redirect(url_for("admin", error="Invalid admin password."))

    return redirect(url_for("admin"))


@app.get("/admin/logout")
def admin_logout():
    session.pop("admin_logged_in", None)
    return redirect(url_for("admin"))


@app.get("/admin")
@app.get("/admin.html")
def admin():
    if not is_admin_logged_in():
        return render_template(
            "admin.html",
            records=None,
            login_mode=True,
            error=request.args.get("error", ""),
        )

    records = prediction_rows()
    return render_template("admin.html", records=records, login_mode=False, error="")


@app.get("/admin/download")
def download_all_predictions():
    blocked = require_admin()
    if blocked:
        return blocked

    csv_text = rows_to_csv(prediction_rows())
    return Response(
        csv_text,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=all_predictions.csv"},
    )


@app.get("/download/<int:record_id>")
def download_prediction(record_id):
    row = prediction_row(record_id)
    if row is None:
        return jsonify({"error": "Prediction record not found."}), 404

    csv_text = rows_to_csv([row])
    return Response(
        csv_text,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=prediction_{record_id}.csv"},
    )


@app.get("/report/<int:record_id>")
def report(record_id):
    row = prediction_row(record_id)
    if row is None:
        return jsonify({"error": "Prediction record not found."}), 404

    return render_template("report.html", **report_context(row))


@app.get("/report.html")
def latest_report():
    rows = prediction_rows()
    if not rows:
        return redirect(url_for("home"))

    return redirect(url_for("report", record_id=rows[0]["id"]))


@app.get("/download/<int:record_id>/word")
def download_word_report(record_id):
    row = prediction_row(record_id)
    if row is None:
        return jsonify({"error": "Prediction record not found."}), 404

    html = render_template("report.html", printable=False, **report_context(row))
    return Response(
        html,
        mimetype="application/msword",
        headers={"Content-Disposition": f"attachment; filename=covid_report_{record_id}.doc"},
    )


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
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
        answers = build_answers(payload)
        used_fallback = False

        try:
            model = load_model()
            model_input = build_model_input(model, answers)
            prediction, risk_probability, confidence = predict(model, model_input)
        except Exception:
            prediction, risk_probability, confidence = fallback_prediction(answers)
            used_fallback = True

        risk_label, recommendation = risk_details(risk_probability)

        try:
            record_id = save_prediction(payload, risk_label, risk_probability, prediction, recommendation)
        except Exception:
            record_id = None

        return jsonify(
            {
                "record_id": record_id,
                "risk_label": risk_label,
                "risk_probability": round(risk_probability, 4),
                "confidence": round(confidence, 4),
                "prediction": prediction,
                "recommendation": recommendation,
                "fallback": used_fallback,
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True)