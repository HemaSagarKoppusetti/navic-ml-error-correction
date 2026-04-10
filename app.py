import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow import keras
from tcn import TCN

st.set_page_config(layout="wide")

st.title("📡 NavIC/GNSS ML Error Correction Dashboard")

# ===============================
# LOAD DATA
# ===============================
RESULTS_FILE = "results/model_comparison_results.csv"
DATA_FILE = "dataset/processed/ml_dataset.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)

data = load_data()
results = pd.read_csv(RESULTS_FILE)
# Features used in training
FEATURES = [
    "mean_snr","sat_count","mean_elevation","mean_azimuth",
    "roll","pitch","yaw","inclination_angle","position_jump"
]
from sklearn.impute import SimpleImputer

X = data[FEATURES]

imputer = SimpleImputer(strategy="mean")
X = pd.DataFrame(
    imputer.fit_transform(X),
    columns=FEATURES
)

scaler = joblib.load("models/scaler.pkl")
X = scaler.transform(X)

# ===============================
# MODEL LOADER (CACHED)
# ===============================

@st.cache_resource
def load_model(model_name):

    if model_name == "Random Forest":
        return (
            joblib.load("models/rf_x.pkl"),
            joblib.load("models/rf_y.pkl")
        )

    elif model_name == "XGBoost":
        return (
            joblib.load("models/xgb_x.pkl"),
            joblib.load("models/xgb_y.pkl")
        )

    elif model_name == "SVM":
        return (
            joblib.load("models/svm_x.pkl"),
            joblib.load("models/svm_y.pkl")
        )

    elif model_name == "LSTM":
        return keras.models.load_model("models/lstm_tuned.h5", compile=False)

    elif model_name == "TCN":
        return keras.models.load_model("models/tcn_model.h5", compile=False)

    return None

# ===============================
# MODEL SELECTOR
# ===============================
st.sidebar.title("⚙️ Model Selection")

model_name = st.sidebar.selectbox(
    "Select Model",
    results["Model"].tolist()
)
selected = results[results["Model"] == model_name].iloc[0]
# ===============================
# MODEL INFORMATION
# ===============================
MODEL_INFO = {
    "Random Forest": {
        "algorithm": "Random Forest Regressor (RF)",
        "context": "An ensemble of decision trees trained on GNSS signal-quality features to predict position error.",
        "why_used": "Works well on tabular data, handles non-linear relationships, and is robust on smaller datasets.",
        "best_for": "Strong baseline for GNSS error correction on structured features.",
    },
    "XGBoost": {
        "algorithm": "Extreme Gradient Boosting Regressor (XGBoost)",
        "context": "A boosting-based tree model that learns error patterns by combining many weak learners.",
        "why_used": "Captures complex feature interactions and usually gives strong performance on tabular ML tasks.",
        "best_for": "Fine-grained error prediction with feature interactions like SNR, elevation, and azimuth.",
    },
    "SVM": {
        "algorithm": "Support Vector Regressor (SVR)",
        "context": "A kernel-based regression model that fits a margin around the target error values.",
        "why_used": "Useful when the relationship is non-linear but the dataset is still manageable in size.",
        "best_for": "Compact regression modeling with scaled features.",
    },
    "LSTM": {
        "algorithm": "Long Short-Term Memory Network (LSTM)",
        "context": "A sequence model designed to learn time-dependent patterns from sliding windows of GNSS features.",
        "why_used": "Used to test whether temporal dependencies improve GNSS error prediction.",
        "best_for": "Sequential GNSS patterns, though in project it performs weaker than tree models.",
    },
    "TCN": {
        "algorithm": "Temporal Convolutional Network (TCN)",
        "context": "A deep sequence model that uses causal convolutions to learn time-series patterns.",
        "why_used": "Evaluated as an advanced temporal model for GNSS signal evolution.",
        "best_for": "Sequence learning, but results show it is not outperforming tree-based models.",
    }
}

info = MODEL_INFO.get(model_name, {})

st.subheader(f"🧠 Model Details : {info.get('algorithm', '-')}")

st.markdown("""
<style>
.card {
    background-color: #111;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #333;
}
</style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="card">
    <h4>Context</h4>
    <p>{info.get("context","-")}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="card">
    <h4>Why Used</h4>
    <p>{info.get("why_used","-")}</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="card">
    <h4>Best For</h4>
    <p>{info.get("best_for","-")}</p>
    </div>
    """, unsafe_allow_html=True)

if model_name in ["LSTM", "TCN"]:
    st.warning(
        "This is a sequence model. In this project, it is evaluated mainly for comparison because the dataset is more tabular than temporal."
    )
# ===============================
# LOAD MODEL
# ===============================
model = load_model(model_name)

# ===============================
# PREDICTION
# ===============================
pred_x, pred_y = None, None

if model_name in ["Random Forest", "XGBoost", "SVM"]:
    pred_x = model[0].predict(X)
    pred_y = model[1].predict(X)

elif model_name in ["LSTM", "TCN"]:
    st.warning("⚠️ Sequence models (LSTM/TCN) require time-series input and are evaluated offline only.")
# ===============================
# CORRECTION
# ===============================
if pred_x is not None:
    data["corrected_x"] = data["x"] - pred_x
    data["corrected_y"] = data["y"] - pred_y

    data["corrected_error"] = np.sqrt(
        data["corrected_x"]**2 + data["corrected_y"]**2
    )

    data["original_error"] = np.sqrt(
        data["x"]**2 + data["y"]**2
    )
else:
    st.info("📊 Showing precomputed metrics for LSTM/TCN")


# ===============================
# MODEL PERFORMANCE (FULL METRICS TABLE)
# ===============================
st.subheader("🏆 Model Performance")

# ---- required columns ----
required_cols = ["Model", "RMSE_X", "RMSE_Y", "MAE_X", "MAE_Y", "R2_X", "R2_Y","Accuracy"]

# ---- check missing columns ----
missing_cols = [col for col in required_cols if col not in results.columns]

if missing_cols:
    st.error(f"Missing columns in results file: {missing_cols}")
else:
    display_df = results[required_cols].copy()

    # ---- round values ----
    for col in required_cols[1:]:
        display_df[col] = display_df[col].round(3)

    # ---- rename for UI ----
    display_df = display_df.rename(columns={
        "RMSE_X": "RMSE X",
        "RMSE_Y": "RMSE Y",
        "MAE_X": "MAE X",
        "MAE_Y": "MAE Y",
        "R2_X": "R² X",
        "R2_Y": "R² Y",
        "Accuracy": "Accuracy"
    })

    # ---- highlight selected model ----
    def highlight_selected(row):
        if row["Model"] == model_name:
            return ["background-color: #1e293b; color: #facc15; font-weight: bold"] * len(row)
        return [""] * len(row)

    styled_df = display_df.style.apply(highlight_selected, axis=1)

    # ---- render ----
    st.dataframe(styled_df, use_container_width=True)

# ===============================
# CONFUSION MATRIX
# ===============================
st.subheader("🔍 Confusion Matrix")

if pred_x is not None:

    true_error = np.sqrt(data["error_x"]**2 + data["error_y"]**2)
    pred_error = np.sqrt(pred_x**2 + pred_y**2)

    threshold = 2.0

    true_class = (true_error < threshold).astype(int)
    pred_class = (pred_error < threshold).astype(int)

    cm = confusion_matrix(true_class, pred_class)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{model_name} Confusion Matrix")

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.pyplot(fig)

else:
    st.info("📊 Confusion matrix not available for LSTM/TCN (sequence models)")

# ===============================
# TRAJECTORY VISUALIZATION
# ===============================
st.subheader("🗺️ Trajectory Comparison")
if pred_x is not None:

    fig = px.line()

    fig.add_scatter(x=data["x"], y=data["y"], name="Raw", line=dict(width=2))

    fig.add_scatter(
        x=data["corrected_x"],
        y=data["corrected_y"],
        name="Corrected",line=dict(width=2)
    )

    st.plotly_chart(fig, width='stretch')

else:
    st.info("🗺️ Trajectory not available for LSTM/TCN")


# ===============================
# ERROR DISTRIBUTION
# ===============================
def apply_model_correction(df, model_name):

    features = [
        "mean_snr",
        "sat_count",
        "mean_elevation",
        "mean_azimuth",
        "roll",
        "pitch",
        "yaw",
        "inclination_angle",
        "position_jump"
    ]

    X = df[features].copy()

    # Handle NaN (VERY IMPORTANT for SVM)
    X = X.fillna(X.mean())

    # -------- LOAD MODEL --------
    if model_name == "Random Forest":
        model_x = joblib.load("models/rf_x.pkl")
        model_y = joblib.load("models/rf_y.pkl")

    elif model_name == "XGBoost":
        model_x = joblib.load("models/xgb_x.pkl")
        model_y = joblib.load("models/xgb_y.pkl")

    elif model_name == "SVM":
        model_x = joblib.load("models/svm_x.pkl")
        model_y = joblib.load("models/svm_y.pkl")

    else:
        return None  # LSTM/TCN skip

    # -------- PREDICT --------
    pred_x = model_x.predict(X)
    pred_y = model_y.predict(X)

    df["pred_error_x"] = pred_x
    df["pred_error_y"] = pred_y

    # -------- CORRECTION --------
    df["corrected_x"] = df["x"] - pred_x
    df["corrected_y"] = df["y"] - pred_y

    # -------- ERROR CALC --------
    df["corrected_error"] = np.sqrt(
        df["corrected_x"]**2 + df["corrected_y"]**2
    )

    df["original_error"] = np.sqrt(
        df["x"]**2 + df["y"]**2
    )

    return df


st.subheader("📊 Error Distribution")

df_corrected = apply_model_correction(data.copy(), model_name)

if df_corrected is not None:

    fig = px.histogram(
        df_corrected,
        x=["original_error", "corrected_error"],
        nbins=50,
        opacity=0.7,
        barmode="overlay",
        title=f"{model_name} Error Distribution"
    )

    fig.update_layout(
        xaxis_title="Error (meters)",
        yaxis_title="Frequency",
        legend_title="Error Type"
    )

    st.plotly_chart(fig, width="stretch")

else:
    st.info("📊 Error distribution not available for LSTM/TCN (sequence models)")
    fig2 = px.histogram(
        data,
        x=["original_error"],
        nbins=50,
        title="Original Error Distribution"
    )

    st.plotly_chart(fig2, width='stretch')

# ===============================
# MODEL COMPARISON (DOUBLE BAR)
# ===============================
st.subheader("📊 Model Comparison")

# reshape data for grouped bar chart
plot_df = results.melt(
    id_vars="Model",
    value_vars=["RMSE_X", "RMSE_Y"],
    var_name="Metric",
    value_name="Value"
)

fig3 = px.bar(
    plot_df,
    x="Model",
    y="Value",
    color="Metric",          # this creates double bars
    barmode="group",         # side-by-side bars
    text_auto=".2f",
    title="RMSE Comparison Across Models",
    color_discrete_map={
    "RMSE_X": "#38bdf8",   # blue
    "RMSE_Y": "#f97316"    # orange
}
)

fig3.update_layout(
    xaxis_title="Model",
    yaxis_title="RMSE (meters)",
    legend_title="Metric"
)

st.plotly_chart(fig3, width="stretch")


mae_df = results.melt(
    id_vars="Model",
    value_vars=["MAE_X", "MAE_Y"],
    var_name="Metric",
    value_name="Value"
)

fig_mae = px.bar(
    mae_df,
    x="Model",
    y="Value",
    color="Metric",
    barmode="group",
    text_auto=".2f",
    color_discrete_map={
        "MAE_X": "#22c55e",   # green
        "MAE_Y": "#eab308"    # yellow
    },
    title="MAE Comparison Across Models"
)

fig_mae.update_layout(
    xaxis_title="Model",
    yaxis_title="MAE (meters)"
)

st.plotly_chart(fig_mae, width="stretch")


r2_df = results.melt(
    id_vars="Model",
    value_vars=["R2_X", "R2_Y"],
    var_name="Metric",
    value_name="Value"
)

fig_r2 = px.bar(
    r2_df,
    x="Model",
    y="Value",
    color="Metric",
    barmode="group",
    text_auto=".2f",
    color_discrete_map={
        "R2_X": "#a855f7",   # purple
        "R2_Y": "#ec4899"    # pink
    },
    title="R² Score Comparison Across Models"
)

fig_r2.update_layout(
    xaxis_title="Model",
    yaxis_title="R² Score"
)

st.plotly_chart(fig_r2, width="stretch")

# ===============================
# INSIGHTS (FIXED + CLEAN UI)
# ===============================
# ===============================
# INSIGHTS (FINAL FIXED VERSION)
# ===============================
st.subheader("🧠 Model Insights")

rmse_x = selected["RMSE_X"]
rmse_y = selected["RMSE_Y"]
r2 = selected["R2_X"]

MODEL_INSIGHTS = {
    "Random Forest": [
        f"Strongest baseline model for this dataset",
        f"RMSE X = {rmse_x:.2f}, RMSE Y = {rmse_y:.2f}",
        f"High R² score = {r2:.2f}, indicating good fit",
        "Handles non-linear GNSS feature relationships effectively",
        "Best suited for structured/tabular NavIC data"
    ],

    "XGBoost": [
        f"High-performance boosting model",
        f"RMSE X = {rmse_x:.2f}, RMSE Y = {rmse_y:.2f}",
        f"R² score = {r2:.2f}",
        "Captures feature interactions (SNR, elevation, azimuth)",
        "Strong alternative to Random Forest"
    ],

    "SVM": [
        f"Kernel-based regression model",
        f"RMSE X = {rmse_x:.2f}, RMSE Y = {rmse_y:.2f}",
        f"R² score = {r2:.2f}",
        "Works well with scaled GNSS features",
        "Slower and less flexible than tree models"
    ],

    "LSTM": [
        f"Sequence-based deep learning model",
        f"RMSE X = {rmse_x:.2f}, RMSE Y = {rmse_y:.2f}",
        f"R² score = {r2:.2f}",
        "Learns temporal patterns from GNSS data",
        "Dataset lacks strong temporal continuity",
        "Underperforms compared to tree-based models"
    ],

    "TCN": [
        f"Temporal Convolutional Network",
        f"RMSE X = {rmse_x:.2f}, RMSE Y = {rmse_y:.2f}",
        f"R² score = {r2:.2f}",
        "Uses convolution-based sequence learning",
        "Not suitable for this dataset structure",
        "Weakest model performance"
    ]
}

points = MODEL_INSIGHTS.get(model_name, ["No insights available"])

# ---------- CLEAN CSS ----------
st.markdown("""
<style>
.insight-card {
    background: linear-gradient(145deg, #020617, #020617);
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #1e293b;
    margin-top: 10px;
}
.insight-title {
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 12px;
}
.insight-item {
    margin-bottom: 8px;
    font-size: 15px;
}
</style>
""", unsafe_allow_html=True)

# ---------- SINGLE CARD ONLY ----------
card_html = f"""
<div class="insight-card">
    <div class="insight-title">📊 {model_name} Performance Summary</div>
    {''.join([f'<div class="insight-item">✔ {p}</div>' for p in points])}
</div>
"""

st.markdown(card_html, unsafe_allow_html=True)

# ===============================
# DATA TABLE
# ===============================
st.subheader("📄 Sample Data")

st.dataframe(data.head(50), width="stretch")