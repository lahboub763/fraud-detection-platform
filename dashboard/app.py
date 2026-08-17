import streamlit as st
import requests


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Fraud Detection Platform",
    page_icon="🛡️",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: gray;
        margin-bottom: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-title">🛡️ Fraud Detection Platform</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered transaction fraud detection</div>',
    unsafe_allow_html=True
)

st.write(
    "Enter transaction information to check whether "
    "the transaction is potentially fraudulent."
)


# ============================================================
# INPUTS
# ============================================================

col1, col2 = st.columns(2)

with col1:

    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=500.0,
        step=10.0
    )

    country = st.selectbox(
        "Country",
        [
            "Italy",
            "France",
            "Spain",
            "Germany",
            "Morocco"
        ]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "card",
            "paypal",
            "bank_transfer"
        ]
    )


with col2:

    device = st.selectbox(
        "Device",
        [
            "desktop",
            "mobile",
            "tablet"
        ]
    )

    hour = st.slider(
        "Transaction Hour",
        min_value=0,
        max_value=23,
        value=12
    )


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "🔍 Check Transaction",
    use_container_width=True
):

    transaction = {
        "amount": amount,
        "country": country,
        "payment_method": payment_method,
        "device": device,
        "hour": hour
    }

    try:

        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=transaction
        )

        if response.status_code == 200:

            result = response.json()

            prediction = result["prediction"]
            probability = result["fraud_probability"]

            st.divider()

            st.subheader("Prediction Result")

            st.metric(
                "Fraud Probability",
                f"{probability * 100:.2f}%"
            )

            st.progress(probability)

            if prediction == "Fraud":

                st.error(
                    "🚨 Fraud Detected"
                )

            else:

                st.success(
                    "✅ Transaction appears legitimate"
                )

        else:

            st.error(
                f"API Error: {response.status_code}"
            )

    except requests.exceptions.ConnectionError:

        st.error(
            "❌ Cannot connect to the Fraud Detection API. "
            "Make sure FastAPI is running."
        )