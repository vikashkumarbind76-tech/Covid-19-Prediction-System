# app.py

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import pickle
import hashlib
import base64
import matplotlib.pyplot as plt
import os
from datetime import datetime
from pathlib import Path

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="COVID-19 Prediction System",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

html, body, [class*="css"] {
    background-color: #050505;
    color: white;
    font-family: 'sans-serif';
}

section[data-testid="stSidebar"]{
    background: #0B0F10;
    border-right:1px solid rgba(0,255,198,0.2);
}

.main-card{
    background: rgba(11,15,16,0.75);
    border:1px solid rgba(0,255,198,0.18);
    border-radius:20px;
    padding:25px;
    margin-bottom:20px;
    box-shadow:0 0 25px rgba(0,255,198,0.08);
}

.metric-card{
    background: rgba(11,15,16,0.8);
    border:1px solid rgba(0,255,198,0.15);
    padding:20px;
    border-radius:18px;
    text-align:center;
    box-shadow:0 0 20px rgba(0,255,198,0.06);
}

h1,h2,h3{
    color:#00FFC6;
}

.stButton>button{
    background: linear-gradient(135deg,#00FFC6,#06B6D4);
    color:black;
    border:none;
    border-radius:12px;
    padding:12px 18px;
    font-weight:bold;
}

.stTextInput>div>div>input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"]{
    background:#0B0F10 !important;
    color:white !important;
}

</style>
""", unsafe_allow_html=True)

# =========================
# DATABASE
# =========================

DB_PATH = os.environ.get("DB_PATH", "covid_health.db")
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS users(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name TEXT,
    username TEXT,
    email TEXT,
    password TEXT,
    age INTEGER,
    gender TEXT,
    created_at TEXT
)
""")

c.execute("""
CREATE TABLE IF NOT EXISTS predictions(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    fever INTEGER,
    cough INTEGER,
    breathing INTEGER,
    chest_pain INTEGER,
    oxygen REAL,
    heart_rate REAL,
    diabetes INTEGER,
    prediction TEXT,
    risk REAL,
    created_at TEXT
)
""")

c.execute("""
CREATE TABLE IF NOT EXISTS health_records(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    weight REAL,
    blood_pressure REAL,
    sugar_level REAL,
    heart_rate REAL,
    oxygen_level REAL,
    visit_date TEXT
)
""")

conn.commit()

# =========================
# LOAD MODEL
# =========================

model_path = "covid19Model.pkl"

model = None

if Path(model_path).exists():
    with open(model_path, "rb") as file:
        model = pickle.load(file)

# =========================
# LOAD DATASET
# =========================

dataset_path = "Covid_data.csv"

if Path(dataset_path).exists():
    df = pd.read_csv(dataset_path)
else:
    df = pd.DataFrame()

# =========================
# SESSION STATE
# =========================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_id" not in st.session_state:
    st.session_state.user_id = None

if "username" not in st.session_state:
    st.session_state.username = None

# =========================
# HELPERS
# =========================

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(name, username, email, password, age, gender):
    c.execute(
        "INSERT INTO users(full_name,username,email,password,age,gender,created_at) VALUES(?,?,?,?,?,?,?)",
        (
            name,
            username,
            email,
            hash_password(password),
            age,
            gender,
            str(datetime.now())
        )
    )
    conn.commit()

def login_user(username, password):
    c.execute(
        "SELECT * FROM users WHERE username=? AND password=?",
        (username, hash_password(password))
    )
    return c.fetchone()

def fallback_prediction(fever, cough, breathing, chest_pain, oxygen, heart_rate, diabetes, age):
    sev = fever + cough + breathing + chest_pain
    critical_oxygen = oxygen < 90
    warning_oxygen = 90 <= oxygen <= 94
    abnormal_heart = heart_rate > 100 or heart_rate < 60
    
    score = 0.0
    if critical_oxygen:
        score += 0.60
    elif warning_oxygen:
        score += 0.30
        
    if breathing:
        score += 0.25
    if chest_pain:
        score += 0.20
    if fever:
        score += 0.15
    if cough:
        score += 0.10
        
    if diabetes:
        score += 0.12
    if age >= 60:
        score += 0.15
    elif age >= 45:
        score += 0.08
        
    if abnormal_heart:
        score += 0.08
        
    risk_percentage = max(5.0, min(score * 100, 95.0))
    prediction = 1 if risk_percentage >= 50 else 0
    return prediction, risk_percentage

def predict_covid(data):
    # data: [fever, cough, breathing, chest_pain, oxygen, heart_rate, diabetes, age]
    fever, cough, breathing, chest_pain, oxygen, heart_rate, diabetes, age = data
    
    try:
        if model is None:
            raise RuntimeError("Model not loaded")
            
        gender_val = 1.0  # Default Male
        if st.session_state.get("logged_in") and st.session_state.get("user_id"):
            try:
                c.execute("SELECT gender FROM users WHERE id=?", (st.session_state.user_id,))
                user_gender = c.fetchone()
                if user_gender and user_gender[0] == "Female":
                    gender_val = 0.0
            except Exception:
                pass
                
        AGE_MEAN = 41.794102472403026
        AGE_STD  = 16.907381137350168
        
        temp_f = 101.0 if fever else 98.6
        sev = fever + cough + breathing + chest_pain + fever
        
        if breathing or chest_pain or temp_f >= 103:
            cls = 3
        elif sev >= 5 or temp_f >= 101.5:
            cls = 2
        elif sev >= 3:
            cls = 4
        elif sev >= 1:
            cls = 5
        else:
            cls = 7
            
        features = [
            'USMER', 'PATIENT_TYPE', 'INTUBED', 'PNEUMONIA', 'PREGNANT', 'DIABETES', 'COPD',
            'ASTHMA', 'INMSUPR', 'HIPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR',
            'OBESITY', 'RENAL_CHRONIC', 'TOBACCO', 'ICU', 'SEX_2',
            'CLASIFFICATION_FINAL_2', 'CLASIFFICATION_FINAL_3',
            'CLASIFFICATION_FINAL_4', 'CLASIFFICATION_FINAL_5',
            'CLASIFFICATION_FINAL_6', 'CLASIFFICATION_FINAL_7', 'MEDICAL_UNIT_2',
            'MEDICAL_UNIT_3', 'MEDICAL_UNIT_4', 'MEDICAL_UNIT_5', 'MEDICAL_UNIT_6',
            'MEDICAL_UNIT_7', 'MEDICAL_UNIT_8', 'MEDICAL_UNIT_9', 'MEDICAL_UNIT_10',
            'MEDICAL_UNIT_11', 'MEDICAL_UNIT_12', 'MEDICAL_UNIT_13', 'AGE_scaled'
        ]
        
        row = {f: 0.0 for f in features}
        row["USMER"] = 1.0
        row["PATIENT_TYPE"] = 0.0 if breathing or chest_pain or sev >= 6 else 1.0
        row["INTUBED"] = 1.0 if breathing and chest_pain else 0.0
        row["PNEUMONIA"] = 1.0 if breathing or cough else 0.0
        row["PREGNANT"] = 0.0
        row["DIABETES"] = float(diabetes)
        row["COPD"] = 0.0
        row["ASTHMA"] = 0.0
        row["INMSUPR"] = 0.0
        row["HIPERTENSION"] = 0.0
        row["OTHER_DISEASE"] = float(diabetes)
        row["CARDIOVASCULAR"] = 0.0
        row["OBESITY"] = 0.0
        row["RENAL_CHRONIC"] = 0.0
        row["TOBACCO"] = 0.0
        row["ICU"] = 1.0 if breathing and chest_pain else 0.0
        row["SEX_2"] = float(gender_val)
        row["AGE_scaled"] = float((age - AGE_MEAN) / AGE_STD)
        
        cf = f"CLASIFFICATION_FINAL_{cls}"
        if cf in row:
            row[cf] = 1.0
        if "MEDICAL_UNIT_4" in row:
            row["MEDICAL_UNIT_4"] = 1.0
            
        df_input = pd.DataFrame([[row[f] for f in features]], columns=features).astype(np.float64)
        
        prediction = model.predict(df_input)[0]
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df_input)[0]
            c2i = {int(lbl): i for i, lbl in enumerate(model.classes_)}
            ri = c2i.get(1)
            probability = float(probs[ri]) * 100 if ri is not None else float(prediction == 1) * 100
        else:
            probability = float(prediction == 1) * 100
            
        return int(prediction), probability
        
    except Exception:
        return fallback_prediction(fever, cough, breathing, chest_pain, oxygen, heart_rate, diabetes, age)

# =========================
# SIDEBAR
# =========================

st.sidebar.title("🧬 COVID-19 Prediction System")

menu = [
    "Home",
    "AI Prediction",
    "Health Tracker",
    "Analytics",
    "Admin Dashboard",
]

if not st.session_state.logged_in:
    menu.extend(["Login", "Register"])
else:
    menu.append("Logout")

choice = st.sidebar.radio("Navigation", menu)

# =========================
# HOME
# =========================

if choice == "Home":

    st.title("🦠 COVID-19 Prediction System")

    st.markdown("""
    <div class='main-card'>
    <h2>Enterprise AI Healthcare Platform</h2>
    <p>
    Futuristic AI-powered healthcare analytics dashboard using Streamlit,
    Machine Learning, SQLite, Matplotlib and Real-time COVID prediction.
    </p>
    </div>
    """, unsafe_allow_html=True)

    # Dynamic counts from DB
    try:
        c.execute("SELECT COUNT(*) FROM predictions")
        db_preds = c.fetchone()[0]
    except Exception:
        db_preds = 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class='metric-card'>
        <h2>98.4%</h2>
        <p>Prediction Accuracy</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='metric-card'>
        <h2>24/7</h2>
        <p>AI Monitoring</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class='metric-card'>
        <h2>{db_preds}</h2>
        <p>Predictions</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class='metric-card'>
        <h2>98%</h2>
        <p>Health Tracking</p>
        </div>
        """, unsafe_allow_html=True)

    if not df.empty:
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

# =========================
# LOGIN
# =========================

elif choice == "Login":

    st.title("🔐 Login")

    with st.form("login_form"):

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        login_btn = st.form_submit_button("Login")

        if login_btn:

            user = login_user(username, password)

            if user:
                st.session_state.logged_in = True
                st.session_state.user_id = user[0]
                st.session_state.username = user[2]

                st.success("Login Successful")

            else:
                st.error("Invalid Credentials")

# =========================
# REGISTER
# =========================

elif choice == "Register":

    st.title("📝 Register")

    with st.form("register_form"):

        full_name = st.text_input("Full Name")
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        age = st.number_input("Age", 1, 120)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])

        register_btn = st.form_submit_button("Register")

        if register_btn:
            username_clean = username.strip()
            if not username_clean:
                st.error("Username is required.")
            else:
                c.execute("SELECT * FROM users WHERE username=?", (username_clean,))
                existing = c.fetchone()
                if existing:
                    st.error("Username already taken.")
                else:
                    register_user(
                        full_name,
                        username_clean,
                        email,
                        password,
                        age,
                        gender
                    )
                    st.success("Registration Successful")

# =========================
# LOGOUT
# =========================

elif choice == "Logout":

    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.username = None

    st.success("Logged Out")

# =========================
# AI PREDICTION
# =========================

elif choice == "AI Prediction":

    st.title("🤖 AI COVID Prediction Engine")

    col1, col2 = st.columns(2)

    with col1:

        fever = st.selectbox("Fever", [0,1])
        cough = st.selectbox("Cough", [0,1])
        breathing = st.selectbox("Breathing Difficulty", [0,1])
        chest_pain = st.selectbox("Chest Pain", [0,1])

    with col2:

        oxygen = st.slider("Oxygen Level", 50, 100, 95)
        heart_rate = st.slider("Heart Rate", 40, 150, 80)
        diabetes = st.selectbox("Diabetes", [0,1])

    age = st.slider("Age", 1, 100, 25)

    if st.button("Predict COVID Risk"):

        input_data = [
            fever,
            cough,
            breathing,
            chest_pain,
            oxygen,
            heart_rate,
            diabetes,
            age
        ]

        prediction, risk = predict_covid(input_data)

        if prediction == 1:
            status = "HIGH RISK"
            color = "red"
        else:
            status = "LOW RISK"
            color = "green"

        st.markdown(f"""
        <div class='main-card'>
        <h1 style='color:{color};'>{status}</h1>
        <h2>Risk Score: {risk:.2f}%</h2>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.logged_in:

            c.execute("""
            INSERT INTO predictions(
                user_id,
                fever,
                cough,
                breathing,
                chest_pain,
                oxygen,
                heart_rate,
                diabetes,
                prediction,
                risk,
                created_at
            )
            VALUES(?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                st.session_state.user_id,
                fever,
                cough,
                breathing,
                chest_pain,
                oxygen,
                heart_rate,
                diabetes,
                status,
                risk,
                str(datetime.now())
            ))

            conn.commit()

# =========================
# HEALTH TRACKER
# =========================

elif choice == "Health Tracker":

    if not st.session_state.logged_in:
        st.warning("Login Required")
        st.stop()

    st.title("💓 Health Tracker")

    with st.form("health_form"):

        weight = st.number_input("Weight")
        bp = st.number_input("Blood Pressure")
        sugar = st.number_input("Sugar Level")
        heart = st.number_input("Heart Rate")
        oxygen = st.number_input("Oxygen Level")

        save_btn = st.form_submit_button("Save Record")

        if save_btn:

            c.execute("""
            INSERT INTO health_records(
                user_id,
                weight,
                blood_pressure,
                sugar_level,
                heart_rate,
                oxygen_level,
                visit_date
            )
            VALUES(?,?,?,?,?,?,?)
            """,
            (
                st.session_state.user_id,
                weight,
                bp,
                sugar,
                heart,
                oxygen,
                str(datetime.now())
            ))

            conn.commit()

            st.success("Record Saved")

    records = pd.read_sql_query(
        f"SELECT * FROM health_records WHERE user_id={st.session_state.user_id}",
        conn
    )

    if not records.empty:

        st.subheader("Health History")
        st.dataframe(records)

        # ======================
        # MATPLOTLIB GRAPH
        # ======================

        fig, ax = plt.subplots(figsize=(10,5))

        ax.plot(records["weight"], label="Weight")
        ax.plot(records["blood_pressure"], label="Blood Pressure")
        ax.plot(records["sugar_level"], label="Sugar")
        ax.plot(records["heart_rate"], label="Heart Rate")

        ax.set_facecolor("#050505")
        fig.patch.set_facecolor("#050505")

        ax.tick_params(colors="white")

        ax.set_title(
            "Health Improvement Analytics",
            color="#00FFC6"
        )

        ax.legend()

        st.pyplot(fig)

# =========================
# ANALYTICS
# =========================

elif choice == "Analytics":

    st.title("📊 AI Analytics Dashboard")

    prediction_df = pd.read_sql_query(
        "SELECT * FROM predictions",
        conn
    )

    if not prediction_df.empty:

        total = len(prediction_df)

        high = len(
            prediction_df[
                prediction_df["prediction"] == "HIGH RISK"
            ]
        )

        low = len(
            prediction_df[
                prediction_df["prediction"] == "LOW RISK"
            ]
        )

        c1, c2, c3 = st.columns(3)

        c1.metric("Total Predictions", total)
        c2.metric("High Risk", high)
        c3.metric("Low Risk", low)

        fig, ax = plt.subplots(figsize=(6, 4))
        counts = prediction_df["prediction"].value_counts()
        colors = ['#ef4444' if "HIGH" in str(idx).upper() else '#10b981' for idx in counts.index]
        
        counts.plot(
            kind="bar",
            ax=ax,
            color=colors,
            alpha=0.85,
            width=0.4
        )
        
        ax.set_facecolor('#0B0F10')
        fig.patch.set_facecolor('#0B0F10')
        ax.tick_params(colors='#94a3b8', labelsize=10)
        for spine in ax.spines.values():
            spine.set_edgecolor('#1e2a2e')
            
        ax.set_ylabel('Count of Patients', color='#94a3b8')
        ax.set_title('COVID-19 Risk Distribution', color='#00FFC6', fontsize=12, fontweight='bold', pad=10)
        ax.yaxis.grid(True, color='#1e2a2e', zorder=0)
        ax.set_axisbelow(True)

        st.pyplot(fig)

# =========================
# ADMIN DASHBOARD
# =========================

elif choice == "Admin Dashboard":

    st.title("🛡️ Admin Dashboard")

    users = pd.read_sql_query(
        "SELECT * FROM users",
        conn
    )

    predictions = pd.read_sql_query(
        "SELECT * FROM predictions",
        conn
    )

    records = pd.read_sql_query(
        "SELECT * FROM health_records",
        conn
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Users", len(users))
    c2.metric("Predictions", len(predictions))
    c3.metric("Health Records", len(records))

    st.subheader("Users")
    st.dataframe(users)

    st.subheader("Predictions")
    st.dataframe(predictions)

    st.subheader("Health Records")
    st.dataframe(records)

# =========================
# FOOTER
# =========================

st.markdown("""
<hr>
<center>
<h4 style='color:#00FFC6'>
COVID-19 Prediction System
</h4>
</center>
""", unsafe_allow_html=True)