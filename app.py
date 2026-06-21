"""
AI Customer Retention Assistant
--------------------------------
Streamlit app that predicts:
  1. CLV Tier (low / medium / high)              -> XGBoost
  2. Churn Risk (low / medium / high)             -> XGBoost
  3. Discount Sensitivity (not_sensitive/sensitive) -> Random Forest
and generates a personalized retention recommendation using the
Gemini API.

None of the three models use avg_discount_pct or days_since_last_order
as features (by design — a product predicting these things shouldn't
require already knowing closely-related answers), so neither field is
collected from the user.

Folder structure expected:
    app.py
    models/
        xgb_clv_model.pkl
        xgb_churn_model.pkl
        rf_discount_model.pkl
        clv_columns.pkl
        churn_columns.pkl
        discount_columns.pkl
    data/
        ecommerce_data.csv
"""

import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime
from google import genai

# ----------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="AI Customer Retention Assistant",
    page_icon="🛍️",
    layout="wide"
)

# ----------------------------------------------------------------------
# CUSTOM CSS
# ----------------------------------------------------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    .subtitle {
        color: #666;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .pred-card {
        background-color: #f7f8fa;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        border: 1px solid #e6e6e6;
        text-align: center;
    }
    .pred-label {
        font-size: 0.85rem;
        color: #777;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .pred-value {
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 0.2rem;
    }
    .tier-high { color: #1d9e75; }
    .tier-medium { color: #d99a1b; }
    .tier-low { color: #c94f4f; }
    .reco-box {
        background-color: #eef4fb;
        border-left: 4px solid #185fa5;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-top: 1rem;
        color: #1a1a1a;
    }
    .reco-box strong {
        color: #0c2d4d;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# LOAD MODELS (cached so they only load once)
# ----------------------------------------------------------------------
@st.cache_resource
def load_models():
    base = "models"
    xgb_clv = joblib.load(os.path.join(base, "xgb_clv_model.pkl"))
    xgb_churn = joblib.load(os.path.join(base, "xgb_churn_model.pkl"))
    rf_discount = joblib.load(os.path.join(base, "rf_discount_model.pkl"))
    clv_columns = joblib.load(os.path.join(base, "clv_columns.pkl"))
    churn_columns = joblib.load(os.path.join(base, "churn_columns.pkl"))
    discount_columns = joblib.load(os.path.join(base, "discount_columns.pkl"))
    return xgb_clv, xgb_churn, rf_discount, clv_columns, churn_columns, discount_columns


@st.cache_data
def load_data():
    return pd.read_csv(os.path.join("data", "ecommerce_data.csv"))


xgb_clv, xgb_churn, rf_discount, clv_columns, churn_columns, discount_columns = load_models()
raw_data = load_data()

CLV_LABELS = {0: "low", 1: "medium", 2: "high"}
DISCOUNT_LABELS = {0: "not_sensitive", 1: "sensitive"}
# churn_risk classes come from LabelEncoder alphabetically: high, low, medium
CHURN_LABELS = {0: "high", 1: "low", 2: "medium"}

# ----------------------------------------------------------------------
# FEATURE ENGINEERING HELPERS
# These mirror exactly what was done in the Colab notebook so that a
# manually entered customer goes through the same transformations.
# ----------------------------------------------------------------------

TOP_COUNTRIES = [
    "US", "IN", "GB", "BR", "DE", "FR", "MX", "AU", "CA", "JP",
    "ES", "NL", "SG", "SE", "PL"
]

PAYMENT_OPTIONS = ["card", "paypal", "wallet", "cod"]
DEVICE_OPTIONS = ["desktop", "mobile", "tablet"]
SOURCE_OPTIONS = ["organic", "direct", "social", "email", "paid", "referral"]
CATEGORY_OPTIONS = [
    "Beauty", "Electronics", "Fashion", "Sports",
    "Home & Kitchen", "Books", "Toys"
]
AGE_GROUP_MAP = {"18-24": 0, "25-34": 1, "35-44": 2, "45-54": 3, "55+": 4}


def build_feature_row(raw: dict) -> pd.DataFrame:
    """
    Takes a dict of raw (human readable) customer attributes and returns
    a single-row, fully encoded DataFrame matching the training pipeline.

    Note: avg_discount_pct and days_since_last_order are intentionally
    NOT included here — none of the three models use them as features.
    """
    row = {}

    # ordinal
    row["age_group"] = AGE_GROUP_MAP[raw["age_group"]]
    row["marketing_opt_in"] = int(raw["marketing_opt_in"])

    # numeric / engineered
    row["total_orders"] = raw["total_orders"]
    row["total_spend_usd"] = raw["total_spend_usd"]
    row["avg_order_value"] = raw["avg_order_value"]
    row["avg_rating_given"] = raw["avg_rating_given"]
    row["total_sessions"] = raw["total_sessions"]
    row["is_repeat_customer"] = int(raw["is_repeat_customer"])
    row["customer_age_days"] = raw["customer_age_days"]
    row["days_since_last_session"] = raw["days_since_last_session"]

    # one-hot: country
    country = raw["country"] if raw["country"] in TOP_COUNTRIES else "Other"
    for c in TOP_COUNTRIES + ["Other"]:
        row[f"country_{c}"] = 1 if country == c else 0

    # one-hot: preferred_payment
    for p in PAYMENT_OPTIONS:
        row[f"preferred_payment_{p}"] = 1 if raw["preferred_payment"] == p else 0

    # one-hot: preferred_device_ord
    for d in DEVICE_OPTIONS:
        row[f"preferred_device_ord_{d}"] = 1 if raw["preferred_device_ord"] == d else 0

    # one-hot: preferred_source
    for s in SOURCE_OPTIONS:
        row[f"preferred_source_{s}"] = 1 if raw["preferred_source"] == s else 0

    # one-hot: top_category_bought
    for cat in CATEGORY_OPTIONS:
        row[f"top_category_bought_{cat}"] = 1 if raw["top_category_bought"] == cat else 0

    return pd.DataFrame([row])


def predict_all(raw: dict):
    """Run all three models on a raw customer dict and return predictions."""
    base_row = build_feature_row(raw)

    X_clv = base_row.reindex(columns=clv_columns, fill_value=0)
    X_churn = base_row.reindex(columns=churn_columns, fill_value=0)
    X_discount = base_row.reindex(columns=discount_columns, fill_value=0)

    clv_pred = xgb_clv.predict(X_clv)[0]
    clv_proba = xgb_clv.predict_proba(X_clv)[0]

    churn_pred = xgb_churn.predict(X_churn)[0]
    churn_proba = xgb_churn.predict_proba(X_churn)[0]

    discount_pred = rf_discount.predict(X_discount)[0]
    discount_proba = rf_discount.predict_proba(X_discount)[0]

    return {
        "clv_tier": CLV_LABELS[clv_pred],
        "clv_confidence": round(max(clv_proba) * 100, 1),
        "churn_risk": CHURN_LABELS[churn_pred],
        "churn_confidence": round(max(churn_proba) * 100, 1),
        "discount_sensitivity": DISCOUNT_LABELS[discount_pred],
        "discount_confidence": round(max(discount_proba) * 100, 1),
    }


# ----------------------------------------------------------------------
# GEMINI API CALL
# ----------------------------------------------------------------------
def get_recommendation(raw: dict, predictions: dict) -> str:
    api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))
    if not api_key:
        return "⚠️ No Gemini API key found. Add GEMINI_API_KEY to Streamlit secrets to enable AI recommendations."

    client = genai.Client(api_key=api_key)

    prompt = f"""You are a customer retention specialist for an e-commerce business.

Customer profile:
- Age group: {raw['age_group']}
- Total orders: {raw['total_orders']}
- Total spend: ${raw['total_spend_usd']}
- Average order value: ${raw['avg_order_value']}
- Total sessions: {raw['total_sessions']}
- Days since last session: {raw['days_since_last_session']}
- Top category: {raw['top_category_bought']}
- Preferred payment: {raw['preferred_payment']}
- Marketing opt-in: {raw['marketing_opt_in']}

Model predictions:
- CLV Tier: {predictions['clv_tier']} ({predictions['clv_confidence']}% confidence)
- Churn Risk: {predictions['churn_risk']} ({predictions['churn_confidence']}% confidence)
- Discount Sensitivity: {predictions['discount_sensitivity']} ({predictions['discount_confidence']}% confidence)

Write a concise, actionable recommendation (under 100 words) for the marketing
team covering: a one-line summary of the situation, the specific action to take,
which channel to use, and whether a discount is appropriate. Be specific and
reference their top category where useful."""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"⚠️ Could not generate recommendation: {e}"


# ----------------------------------------------------------------------
# UI HELPERS
# ----------------------------------------------------------------------
def render_predictions(predictions: dict):
    col1, col2, col3 = st.columns(3)

    with col1:
        cls = {"low": "tier-low", "medium": "tier-medium", "high": "tier-high"}[predictions["clv_tier"]]
        st.markdown(f"""
        <div class="pred-card">
            <div class="pred-label">CLV Tier</div>
            <div class="pred-value {cls}">{predictions['clv_tier'].title()}</div>
            <div style="color:#999; font-size:0.8rem;">{predictions['clv_confidence']}% confidence</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        cls = {"low": "tier-high", "medium": "tier-medium", "high": "tier-low"}[predictions["churn_risk"]]
        st.markdown(f"""
        <div class="pred-card">
            <div class="pred-label">Churn Risk</div>
            <div class="pred-value {cls}">{predictions['churn_risk'].title()}</div>
            <div style="color:#999; font-size:0.8rem;">{predictions['churn_confidence']}% confidence</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        is_sensitive = predictions["discount_sensitivity"] == "sensitive"
        cls = "tier-medium" if is_sensitive else "tier-high"
        label = "Sensitive" if is_sensitive else "Not Sensitive"
        st.markdown(f"""
        <div class="pred-card">
            <div class="pred-label">Discount Sensitivity</div>
            <div class="pred-value {cls}">{label}</div>
            <div style="color:#999; font-size:0.8rem;">{predictions['discount_confidence']}% confidence</div>
        </div>
        """, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------
st.markdown('<p class="main-title">🛍️ AI Customer Retention Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict customer value, churn risk, and discount sensitivity — then get an AI-generated retention strategy.</p>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔍 Existing Customer Lookup", "📝 New Customer"])

# ----------------------------------------------------------------------
# TAB 1 — EXISTING CUSTOMER LOOKUP
# ----------------------------------------------------------------------
with tab1:
    st.write("Search for a customer from the existing database to see their predicted profile.")

    name_options = raw_data["name"].dropna().unique().tolist()
    selected_name = st.selectbox("Select a customer", options=sorted(name_options))

    if st.button("Run Prediction", key="lookup_btn"):
        customer_row = raw_data[raw_data["name"] == selected_name].iloc[0]

        raw = {
            "age_group": customer_row["age_group"],
            "marketing_opt_in": bool(customer_row["marketing_opt_in"]),
            "total_orders": int(customer_row["total_orders"]),
            "total_spend_usd": float(customer_row["total_spend_usd"]),
            "avg_order_value": float(customer_row["avg_order_value"]),
            "avg_rating_given": float(customer_row["avg_rating_given"]) if pd.notna(customer_row["avg_rating_given"]) else 3.5,
            "total_sessions": int(customer_row["total_sessions"]),
            "is_repeat_customer": bool(customer_row["is_repeat_customer"]),
            "customer_age_days": (datetime.now() - pd.to_datetime(customer_row["signup_date"])).days,
            "days_since_last_session": (datetime.now() - pd.to_datetime(customer_row["last_session_date"])).days if pd.notna(customer_row["last_session_date"]) else 9999,
            "country": customer_row["country"],
            "preferred_payment": customer_row["preferred_payment"],
            "preferred_device_ord": customer_row["preferred_device_ord"],
            "preferred_source": customer_row["preferred_source"],
            "top_category_bought": customer_row["top_category_bought"],
        }

        predictions = predict_all(raw)

        st.subheader(f"Results for {selected_name}")
        render_predictions(predictions)

        with st.spinner("Generating AI recommendation..."):
            recommendation = get_recommendation(raw, predictions)

        st.markdown(f"""
        <div class="reco-box">
            <strong>🤖 AI Recommendation</strong><br><br>{recommendation}
        </div>
        """, unsafe_allow_html=True)

# ----------------------------------------------------------------------
# TAB 2 — NEW CUSTOMER FORM
# ----------------------------------------------------------------------
with tab2:
    st.write("Enter a customer's profile manually to predict their segment in real time.")

    c1, c2, c3 = st.columns(3)

    with c1:
        age_group = st.selectbox("Age Group", list(AGE_GROUP_MAP.keys()))
        total_orders = st.number_input("Total Orders", min_value=0, value=3)
        total_spend_usd = st.number_input("Total Spend (USD)", min_value=0.0, value=250.0)
        avg_order_value = st.number_input("Avg Order Value (USD)", min_value=0.0, value=80.0)

    with c2:
        total_sessions = st.number_input("Total Sessions", min_value=0, value=10)
        days_since_last_session = st.number_input("Days Since Last Session", min_value=0, value=20)
        customer_age_days = st.number_input("Customer Age (days since signup)", min_value=0, value=400)
        avg_rating_given = st.slider("Avg Rating Given", 1.0, 5.0, 4.0)

    with c3:
        country = st.selectbox("Country", TOP_COUNTRIES + ["Other"])
        preferred_payment = st.selectbox("Preferred Payment", PAYMENT_OPTIONS)
        preferred_device_ord = st.selectbox("Preferred Device", DEVICE_OPTIONS)
        preferred_source = st.selectbox("Preferred Source", SOURCE_OPTIONS)
        top_category_bought = st.selectbox("Top Category", CATEGORY_OPTIONS)

    c4, c5 = st.columns(2)
    with c4:
        marketing_opt_in = st.checkbox("Marketing Opt-in", value=True)
    with c5:
        is_repeat_customer = st.checkbox("Repeat Customer", value=True)

    if st.button("Predict", key="form_btn"):
        raw = {
            "age_group": age_group,
            "marketing_opt_in": marketing_opt_in,
            "total_orders": total_orders,
            "total_spend_usd": total_spend_usd,
            "avg_order_value": avg_order_value,
            "avg_rating_given": avg_rating_given,
            "total_sessions": total_sessions,
            "is_repeat_customer": is_repeat_customer,
            "customer_age_days": customer_age_days,
            "days_since_last_session": days_since_last_session,
            "country": country,
            "preferred_payment": preferred_payment,
            "preferred_device_ord": preferred_device_ord,
            "preferred_source": preferred_source,
            "top_category_bought": top_category_bought,
        }

        predictions = predict_all(raw)

        st.subheader("Prediction Results")
        render_predictions(predictions)

        with st.spinner("Generating AI recommendation..."):
            recommendation = get_recommendation(raw, predictions)

        st.markdown(f"""
        <div class="reco-box">
            <strong>🤖 AI Recommendation</strong><br><br>{recommendation}
        </div>
        """, unsafe_allow_html=True)
