import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# Page Setup

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="wide"
)


# CSS

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #1f2937 100%);
    }

    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #f9fafb !important;
    }

    section[data-testid="stSidebar"] input {
        background-color: #ffffff !important;
        color: #111827 !important;
    }

    section[data-testid="stSidebar"] [data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #111827 !important;
        border-radius: 10px !important;
    }

    section[data-testid="stSidebar"] [data-baseweb="select"] span {
        color: #111827 !important;
    }

    section[data-testid="stSidebar"] [data-baseweb="select"] svg {
        fill: #111827 !important;
        color: #111827 !important;
        stroke: #111827 !important;
    }

    section[data-testid="stSidebar"] [data-testid="stNumberInput"] button {
        background-color: #ffffff !important;
        color: #111827 !important;
        border-color: #e5e7eb !important;
    }

    section[data-testid="stSidebar"] [data-testid="stNumberInput"] button svg {
        fill: #111827 !important;
        color: #111827 !important;
        stroke: #111827 !important;
    }

    section[data-testid="stSidebar"] .stButton > button,
    section[data-testid="stSidebar"] [data-testid="stFormSubmitButton"] button {
        background-color: #ef4444 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        height: 48px !important;
    }

    section[data-testid="stSidebar"] .stButton > button:hover,
    section[data-testid="stSidebar"] [data-testid="stFormSubmitButton"] button:hover {
        background-color: #dc2626 !important;
        color: #ffffff !important;
    }

    .main-title {
        font-size: 46px;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 28px;
    }

    .metric-card {
        border: 1px solid #dbe3ef;
        border-radius: 14px;
        padding: 24px;
        background-color: #ffffff;
        min-height: 130px;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.04);
    }

    .metric-label {
        font-size: 15px;
        color: #475569;
        margin-bottom: 10px;
    }

    .metric-value {
        font-size: 38px;
        font-weight: 750;
        color: #111827;
    }

    .section-line {
        border-top: 1px solid #d1d5db;
        margin: 36px 0;
    }

    .result-safe {
        padding: 16px 20px;
        border-radius: 12px;
        background-color: #dcfce7;
        color: #166534;
        font-weight: 700;
        font-size: 18px;
        margin-top: 18px;
    }

    .result-fraud {
        padding: 16px 20px;
        border-radius: 12px;
        background-color: #fee2e2;
        color: #991b1b;
        font-weight: 700;
        font-size: 18px;
        margin-top: 18px;
    }

    .empty-box {
        border: 1px dashed #cbd5e1;
        border-radius: 14px;
        padding: 32px;
        background-color: #ffffff;
        color: #475569;
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Load Model

@st.cache_resource
def load_model():
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "best_fraud_model.pkl"

    with open(model_path, "rb") as file:
        model_package = pickle.load(file)

    return model_package


model_package = load_model()
model = model_package["model"]
threshold = float(model_package["threshold"])


# Session State

if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

if "input_df" not in st.session_state:
    st.session_state.input_df = None

if "risk_score" not in st.session_state:
    st.session_state.risk_score = 0.0

if "prediction" not in st.session_state:
    st.session_state.prediction = "Safe"

if "distance_km" not in st.session_state:
    st.session_state.distance_km = 0.0

if "amount" not in st.session_state:
    st.session_state.amount = 100.0

if "transaction_hour" not in st.session_state:
    st.session_state.transaction_hour = 14


# Helper Functions

def calculate_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return 6371 * c


def predict_transaction(input_df):
    risk_score = float(model.predict_proba(input_df)[0][1])
    prediction = "Fraud" if risk_score >= threshold else "Safe"

    return risk_score, prediction


def create_input_dataframe(
    amt,
    category,
    gender,
    state,
    job,
    city_pop,
    age,
    transaction_hour,
    transaction_day,
    customer_lat,
    customer_long,
    merchant_lat,
    merchant_long
):
    distance_km = calculate_distance(
        customer_lat,
        customer_long,
        merchant_lat,
        merchant_long
    )

    is_weekend = 1 if transaction_day in [5, 6] else 0
    amount_log = np.log1p(amt)

    input_df = pd.DataFrame([{
        "amt": amt,
        "amount_log": amount_log,
        "city_pop": city_pop,
        "transaction_hour": transaction_hour,
        "transaction_day": transaction_day,
        "is_weekend": is_weekend,
        "age": age,
        "distance_km": distance_km,
        "category": category,
        "gender": gender,
        "state": state,
        "job": job
    }])

    return input_df, distance_km


def create_radar_chart(amount, distance, hour, risk_score):
    late_night_risk = 100 if hour >= 22 or hour <= 5 else 25

    factors = {
        "Transaction<br>Amt": min(amount / 1000 * 100, 100),
        "Distance (km)": min(distance / 500 * 100, 100),
        "Late Hour": late_night_risk,
        "Risk Score": risk_score * 100
    }

    labels = list(factors.keys())
    values = list(factors.values())

    labels.append(labels[0])
    values.append(values[0])

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=labels,
        fill="toself",
        line_color="#2563eb",
        fillcolor="rgba(37, 99, 235, 0.22)"
    ))

    fig.update_layout(
        # This section controls the overall text color
        font=dict(
            family="Inter, sans-serif",
            size=14,
            color="#1e293b"  # Dark Slate Blue for better contrast
        ),
        polar=dict(
            bgcolor="white", # Makes the chart "pop" against the page background
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=11, color="#475569"), # Small numbers color
                gridcolor="#e2e8f0" # Lighten the grid lines
            ),
            angularaxis=dict(
                tickfont=dict(size=13, color="#0f172a", weight="bold"), # Labels (Distance, Risk, etc.)
                rotation=90, # Rotates the chart for better label placement
                direction="clockwise",
                gridcolor="#e2e8f0"
            )
        ),
        showlegend=False,
        height=400, # Increased slightly for label breathing room
        margin=dict(l=60, r=60, t=40, b=40), # Increased margins so text doesn't cut off
        paper_bgcolor="rgba(0,0,0,0)", # Transparent to match your app background
        plot_bgcolor="rgba(0,0,0,0)"
    )

    return fig


# Options

state_name_to_code = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY"
}

category_options = [
    "grocery_pos",
    "shopping_net",
    "shopping_pos",
    "gas_transport",
    "misc_net",
    "misc_pos",
    "food_dining",
    "entertainment",
    "kids_pets",
    "personal_care",
    "health_fitness",
    "travel",
    "home"
]

job_options = [
    "Accountant",
    "Architect",
    "Attorney",
    "Doctor",
    "Engineer",
    "Financial adviser",
    "Manager",
    "Nurse",
    "Programmer",
    "Sales executive",
    "Scientist",
    "Teacher",
    "Technician",
    "Writer",
    "Other"
]

day_options = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}


# Sidebar Form

with st.sidebar:
    st.markdown("## Sentinel AI")
    st.caption("Enter transaction parameters below:")

    with st.form("prediction_form"):
        amt = st.number_input(
            "Amount ($)",
            min_value=0.0,
            value=4324.0,
            step=10.0
        )

        category = st.selectbox(
            "Merchant Category",
            category_options,
            index=category_options.index("food_dining")
        )

        transaction_hour = st.slider(
            "Hour (24h)",
            min_value=0,
            max_value=23,
            value=18
        )

        selected_day = st.selectbox(
            "Day",
            list(day_options.keys()),
            index=list(day_options.keys()).index("Thursday")
        )

        st.divider()

        age = st.slider(
            "Customer Age",
            min_value=18,
            max_value=100,
            value=35
        )

        gender = st.radio(
            "Gender",
            ["M", "F"],
            horizontal=True
        )

        selected_state_name = st.selectbox(
            "State",
            list(state_name_to_code.keys()),
            index=list(state_name_to_code.keys()).index("California")
        )

        job = st.selectbox(
            "Job",
            job_options,
            index=job_options.index("Engineer")
        )

        city_pop = st.number_input(
            "City Population",
            min_value=0,
            value=50000,
            step=1000
        )

        st.divider()

        customer_lat = st.number_input(
            "Customer Latitude",
            value=34.0522,
            format="%.6f"
        )

        customer_long = st.number_input(
            "Customer Longitude",
            value=-118.2437,
            format="%.6f"
        )

        merchant_lat = st.number_input(
            "Merchant Latitude",
            value=34.1000,
            format="%.6f"
        )

        merchant_long = st.number_input(
            "Merchant Longitude",
            value=-118.3000,
            format="%.6f"
        )

        predict_clicked = st.form_submit_button(
            "Predict Fraud Risk",
            use_container_width=True
        )


# Prediction

if predict_clicked:
    state = state_name_to_code[selected_state_name]
    transaction_day = day_options[selected_day]

    input_df, distance_km = create_input_dataframe(
        amt,
        category,
        gender,
        state,
        job,
        city_pop,
        age,
        transaction_hour,
        transaction_day,
        customer_lat,
        customer_long,
        merchant_lat,
        merchant_long
    )

    risk_score, prediction = predict_transaction(input_df)

    st.session_state.prediction_done = True
    st.session_state.input_df = input_df
    st.session_state.risk_score = risk_score
    st.session_state.prediction = prediction
    st.session_state.distance_km = distance_km
    st.session_state.amount = amt
    st.session_state.transaction_hour = transaction_hour


# Main Dashboard

st.markdown(
    '<div class="main-title">Financial Integrity Monitor</div>',
    unsafe_allow_html=True
)

if not st.session_state.prediction_done:
    st.markdown(
        """
        <div class="empty-box">
            Enter the transaction details in the left panel and click
            <b>Predict Fraud Risk</b> to view the risk analysis.
        </div>
        """,
        unsafe_allow_html=True
    )

else:
    risk_score = st.session_state.risk_score
    prediction = st.session_state.prediction
    distance_km = st.session_state.distance_km
    amt = st.session_state.amount
    transaction_hour = st.session_state.transaction_hour
    input_df = st.session_state.input_df

    progress_value = float(max(0.0, min(risk_score, 1.0)))

    card1, card2 = st.columns(2)
    # Risk Card
    with card1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Risk Probability</div>
                <div class="metric-value">{risk_score * 100:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    # Distance Card
    with card2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Spatial Gap</div>
                <div class="metric-value">{distance_km:.2f} km</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # DYNAMIC COLOR PROGRESS BAR
    bar_color = "#22c55e"  # Green
    if risk_score > 0.5 and risk_score < threshold:
        bar_color = "#f59e0b"  # Amber/Orange
    elif risk_score >= threshold:
        bar_color = "#ef4444"  # Red

    st.markdown(f"""
        <style>
            div[data-testid="stProgress"] > div > div > div > div {{
                background-color: {bar_color} !important;
            }}
        </style>""", unsafe_allow_html=True)
    st.progress(progress_value)

    # LOGIC
    if prediction == "Fraud":
        st.markdown(
            '<div class="result-fraud">🚨 CRITICAL: This transaction is flagged as Fraud and should be blocked.</div>',
            unsafe_allow_html=True
        )
    elif risk_score > 0.4:
        st.markdown(
            f'<div style="padding: 16px 20px; border-radius: 12px; background-color: #fef3c7; color: #92400e; font-weight: 700; font-size: 18px; margin-top: 18px;">'
            f'⚠️ CAUTION: High risk score ({risk_score*100:.1f}%), Review suggested.</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="result-safe">✅ VERIFIED: This transaction appears safe based on current patterns.</div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)

    left_col, right_col = st.columns(2, gap="large")

    # What If Score
    with left_col:
        st.subheader("What-If Stress Test")
        st.write("Check how changing the transaction amount affects the fraud risk.")

        new_amount = st.slider(
            "Simulate different amount ($)",
            min_value=0.0,
            max_value=5000.0,
            value=min(float(amt), 5000.0),
            step=10.0
        )

        what_if_df = input_df.copy()
        what_if_df["amt"] = new_amount
        what_if_df["amount_log"] = np.log1p(new_amount)

        what_if_score, what_if_prediction = predict_transaction(what_if_df)

        wi_bar_color = "#22c55e"
        if what_if_score > 0.4 and what_if_score < threshold:
            wi_bar_color = "#f59e0b"
        elif what_if_score >= threshold:
            wi_bar_color = "#ef4444"

        st.markdown(f"""
            <style>
                .stProgress > div > div > div > div {{
                    background-color: {wi_bar_color} !important;
                }}
            </style>""", unsafe_allow_html=True)
        
        st.progress(float(max(0.0, min(what_if_score, 1.0))))

        st.markdown(
            f"At **${new_amount:,.2f}**, the risk becomes **{what_if_score * 100:.2f}%**."
        )

        if what_if_score >= threshold:
            st.error("Simulation Result: Fraud Threshold Exceeded.")
        elif what_if_score > 0.4:
            st.markdown("""
                <div style="
                    background-color:#fef3c7;
                    color:#92400e;
                    padding:14px;
                    border-radius:10px;
                    font-weight:400;
                ">
                    Simulation Result: Elevated Risk detected.
                </div>
            """, unsafe_allow_html=True)

        else:
            st.info("Simulation Result: Safe profile maintained.")

    with right_col:
        st.subheader("Risk Vector Analysis")
        radar_fig = create_radar_chart(amt, distance_km, transaction_hour, risk_score)
        st.plotly_chart(radar_fig, use_container_width=True)