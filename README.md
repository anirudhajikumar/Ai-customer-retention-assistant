# 🛍️ AI Customer Retention Assistant

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)](https://xgboost.readthedocs.io/)
[![Gemini](https://img.shields.io/badge/Google%20GenAI-Gemini%202.5-orange.svg)](https://deepmind.google/technologies/gemini/)

Retaining customers is far more cost-effective than acquiring new ones. The **AI Customer Retention Assistant** is a modern, data-driven tool designed to help e-commerce marketing teams identify at-risk customers, understand their value, and instantly generate personalized, AI-driven retention strategies.

This project combines robust machine learning predictions (using **XGBoost** and **Random Forest**) with state-of-the-art Generative AI (**Gemini 2.5 Flash**) to turn raw predictive analytics into immediate, human-readable action plans.

---

## 🎯 Key Capabilities

*   **Three-Dimensional Prediction Engine:**
    *   **Customer Lifetime Value (CLV) Tier:** Predicts whether a customer falls into a `High`, `Medium`, or `Low` value tier (XGBoost).
    *   **Churn Risk:** Classifies customer churn probability as `High`, `Medium`, or `Low` (XGBoost).
    *   **Discount Sensitivity:** Identifies if a customer is likely to respond to discounts (`Sensitive` vs `Not Sensitive`), preventing unnecessary margin erosion (Random Forest).
*   **Generative AI Action Plans:** Instantly translates model outputs and customer histories into concise, personalized marketing recommendations (under 100 words) using the **Google GenAI SDK**.
*   **Dual-Mode Interface:**
    *   **Existing Customer Lookup:** Search and analyze customers already present in the database.
    *   **Real-time Sandbox:** Input a hypothetical customer profile manually to instantly test scenario outputs.

---

## 🛠️ Tech Stack & Architecture

*   **Frontend UI:** [Streamlit](https://streamlit.io/) (for a clean, responsive, and interactive dashboard)
*   **Data Processing:** [Pandas](https://pandas.pydata.org/) & [Scikit-learn](https://scikit-learn.org/)
*   **Machine Learning Models:**
    *   **XGBoost Classifier** (for CLV and Churn Risk)
    *   **Random Forest Classifier** (for Discount Sensitivity)
*   **Generative AI:** [Google GenAI SDK](https://github.com/google/generative-ai-python) (`gemini-2.5-flash`)
*   **Serialization:** [Joblib](https://joblib.readthedocs.io/) (for model and feature-column persistence)

---

## 🚀 Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/anirudhajikumar/Ai-customer-retention-assistant.git
cd Ai-customer-retention-assistant
```

### 2. Set Up a Virtual Environment & Install Dependencies
Create a virtual environment (optional but recommended) and install the required libraries:
```bash
# Using standard venv
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Set Up Your Gemini API Key
The assistant uses the Gemini API to formulate marketing recommendations. Get your API key from [Google AI Studio](https://aistudio.google.com/).

You can set the key in either of these two ways:

#### Option A: Streamlit Secrets (Recommended)
Create a directory named `.streamlit` at the root of the project, add a `secrets.toml` file, and add your API key:
```toml
# .streamlit/secrets.toml
GEMINI_API_KEY = "your_actual_api_key_here"
```

#### Option B: Environment Variable
Alternatively, set the key as an environment variable in your terminal:
```bash
# On Linux/macOS
export GEMINI_API_KEY="your_actual_api_key_here"

# On Windows (Command Prompt)
set GEMINI_API_KEY="your_actual_api_key_here"

# On Windows (PowerShell)
$env:GEMINI_API_KEY="your_actual_api_key_here"
```

### 4. Run the Streamlit Dashboard
Launch the web application locally:
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 📂 Project Structure

```
├── .devcontainer/         # Dev container configuration
├── data/
│   └── ecommerce_data.csv # E-commerce customer database
├── models/
│   ├── xgb_clv_model.pkl       # CLV tier classification model
│   ├── xgb_churn_model.pkl     # Churn risk classification model
│   ├── rf_discount_model.pkl   # Discount sensitivity model
│   ├── clv_columns.pkl         # Expected feature columns for CLV
│   ├── churn_columns.pkl       # Expected feature columns for Churn
│   └── discount_columns.pkl    # Expected feature columns for Discount
├── app.py                 # Main Streamlit web application
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

---

## 🧠 Business Logic & Feature Design

To ensure model predictions are highly robust and don't rely on trivial features, **none of the models use `avg_discount_pct` or `days_since_last_order` during training or inference**.

Instead, features are built from underlying behaviors:
*   Demographics & Sign-up Details (Age group, Country, Customer age in days)
*   Interaction History (Total sessions, Days since last session, Average rating given)
*   Transaction History (Total orders, Total spend in USD, Average order value, Top category bought)
*   Engagement Prefs (Marketing opt-in status, Preferred device, Preferred payment method, Traffic source)
