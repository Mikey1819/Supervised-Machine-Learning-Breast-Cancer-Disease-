import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# ==============================
# Config + Data
# ==============================
st.set_page_config(page_title="Breast Cancer Diagnosis", layout="wide")

DATA_PATH = "/breast_cancer.csv"
df_raw = pd.read_csv(DATA_PATH)

# Clean columns if present
df = df_raw.drop(columns=["Unnamed: 32"], errors="ignore").copy()

# Encode target
if df["diagnosis"].dtype != np.number:
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

# Split features/target
X = df.drop(columns=["id", "diagnosis"], errors="ignore")
y = df["diagnosis"].astype(int)

# Hold min/max/mean for sliders
feature_min = X.min()
feature_max = X.max()
feature_mean = X.mean()

# Train/test split + scale
test_size = 0.2
random_state = 42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ==============================
# Helpers
# ==============================
@st.cache_data
def top_features_by_corr(X_df: pd.DataFrame, y_series: pd.Series, top_n: int = 8):
    """
    Compute absolute point-biserial correlation (Pearson between feature and binary y).
    Returns top_n feature names.
    """
    corr = {}
    y_float = y_series.astype(float)
    for col in X_df.columns:
        # Handle constant columns
        if X_df[col].std() == 0:
            corr[col] = 0.0
        else:
            corr[col] = abs(np.corrcoef(X_df[col].values, y_float.values)[0, 1])
    ranked = sorted(corr.items(), key=lambda kv: kv[1], reverse=True)
    return [k for k, _ in ranked[:top_n]], pd.Series(corr).sort_values(ascending=False)

def train_all_models(Xtr, ytr):
    models = {
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "SVM (linear)": SVC(kernel="linear", probability=True)
    }
    for m in models.values():
        m.fit(Xtr, ytr)
    return models

def evaluate_models(models, Xte, yte):
    rows = []
    y_preds = {}
    for name, model in models.items():
        yp = model.predict(Xte)
        y_preds[name] = yp
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(yte, yp),
            "Precision": precision_score(yte, yp, zero_division=0),
            "Recall": recall_score(yte, yp, zero_division=0),
            "F1": f1_score(yte, yp, zero_division=0)
        })
    return pd.DataFrame(rows).set_index("Model"), y_preds

def plot_confusion(cm, labels=("Benign", "Malignant"), title="Confusion Matrix"):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    st.pyplot(fig)

# ==============================
# Train + Evaluate
# ==============================
models = train_all_models(X_train_s, y_train)
metrics_df, y_preds = evaluate_models(models, X_test_s, y_test)

# ==============================
# UI - Tabs
# ==============================
tab_data, tab_perf, tab_pred = st.tabs(["📊 Data", "🤖 Model Performance", "🔮 Predict"])

# ------------------ DATA TAB ------------------
with tab_data:
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Dataset Preview")
        st.dataframe(df_raw, use_container_width=True, height=360)
    with c2:
        st.subheader("Summary")
        st.write("Rows:", df.shape[0])
        st.write("Features:", X.shape[1])
        st.write("Class balance:")
        st.bar_chart(df["diagnosis"].map({0: "Benign", 1: "Malignant"}).value_counts())

    st.markdown("---")
    st.subheader("Quick Explorers")
    colA, colB, colC = st.columns(3)
    with colA:
        feat_hist = st.selectbox("Histogram feature", X.columns, index=0, key="hist")
        fig, ax = plt.subplots()
        sns.histplot(X[feat_hist], bins=30, kde=True, ax=ax)
        ax.set_title(f"Histogram: {feat_hist}")
        st.pyplot(fig)
    with colB:
        x_sc = st.selectbox("Scatter X", X.columns, index=0, key="sx")
        y_sc = st.selectbox("Scatter Y", X.columns, index=1, key="sy")
        fig, ax = plt.subplots()
        cmap = df["diagnosis"].map({0: "tab:blue", 1: "tab:orange"})
        ax.scatter(df[x_sc], df[y_sc], s=12, c=cmap)
        ax.set_xlabel(x_sc); ax.set_ylabel(y_sc); ax.set_title("Scatter by Diagnosis")
        st.pyplot(fig)
    with colC:
        feat_box = st.selectbox("Boxplot feature", X.columns, index=2, key="box")
        fig, ax = plt.subplots()
        tmp = pd.DataFrame({"diagnosis": df["diagnosis"].map({0: "Benign", 1: "Malignant"}),
                            feat_box: df[feat_box]})
        sns.boxplot(data=tmp, x="diagnosis", y=feat_box, ax=ax)
        ax.set_title(f"Boxplot: {feat_box} by Diagnosis")
        st.pyplot(fig)

# ------------------ PERFORMANCE TAB ------------------
with tab_perf:
    st.subheader("Model Performance Comparison")
    st.dataframe((metrics_df * 100).round(2))

    # Bar chart of metrics
    st.markdown("**Bar Chart (higher is better)**")
    metric_to_plot = st.selectbox("Metric", ["Accuracy", "Precision", "Recall", "F1"], index=0)
    fig, ax = plt.subplots()
    ax.bar(metrics_df.index, metrics_df[metric_to_plot] * 100)
    ax.set_ylabel(f"{metric_to_plot} (%)")
    ax.set_ylim(0, 100)
    for idx, v in enumerate(metrics_df[metric_to_plot] * 100):
        ax.text(idx, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Confusion Matrices")
    cm_cols = st.columns(3)
    for i, (name, yp) in enumerate(y_preds.items()):
        with cm_cols[i]:
            cm = confusion_matrix(y_test, yp)
            plot_confusion(cm, title=name)

    st.markdown("---")
    st.subheader("Feature Importance")
    st.caption("Logistic Regression coefficients (absolute value) and optional Random Forest importances.")

    # Logistic Regression coefficients
    if hasattr(models["Logistic Regression"], "coef_"):
        coefs = np.abs(models["Logistic Regression"].coef_[0])
        imp_lr = pd.Series(coefs, index=X.columns).sort_values(ascending=False)
        st.write("**Top 15 (Logistic Regression | abs(coef))**")
        fig, ax = plt.subplots(figsize=(8, 5))
        imp_lr.head(15).plot(kind="bar", ax=ax)
        ax.set_ylabel("Importance")
        st.pyplot(fig)

    # Optional Random Forest for non-linear importance (trained only for importances)
    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(X_train, y_train)  # NOTE: tree models don't need scaling
    imp_rf = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    st.write("**Top 15 (Random Forest | Gini Importance)**")
    fig, ax = plt.subplots(figsize=(8, 5))
    imp_rf.head(15).plot(kind="bar", ax=ax)
    ax.set_ylabel("Importance")
    st.pyplot(fig)

# ------------------ PREDICT TAB ------------------
with tab_pred:
    st.subheader("Make a Prediction")

    mode = st.radio("Input method",
                    ["Pick a row from dataset", "Use sliders (top features)", "Use sliders (all features)"],
                    index=0)

    # Precompute top features by correlation
    topN_default = 8
    top_feats, corr_series = top_features_by_corr(X, y, top_n=topN_default)

    if mode == "Pick a row from dataset":
        # 1-based row number for user friendliness: first row = 1
        row_num = st.number_input(f"Row number (1 – {len(df)})", min_value=1, max_value=len(df), value=1, step=1)
        row = X.iloc[row_num - 1:row_num]
        st.write("Selected row:")
        st.dataframe(row)
        X_in = row.values

    elif mode == "Use sliders (top features)":
        st.caption(f"Only the **top {topN_default} most correlated** features with the target are shown. Others are auto-filled with dataset means.")
        chosen = st.multiselect("Choose top features to adjust", top_feats, default=top_feats)
        values = {}
        for col in chosen:
            values[col] = st.slider(
                col,
                float(feature_min[col]), float(feature_max[col]),
                float(feature_mean[col])
            )
        # Build full feature vector: chosen get slider value, others mean
        full = []
        for col in X.columns:
            full.append(values[col] if col in values else float(feature_mean[col]))
        X_in = np.array(full, dtype=float).reshape(1, -1)

    else:  # sliders all features
        st.caption("Adjust any feature; defaults are set to column means.")
        values_all = {}
        for col in X.columns:
            values_all[col] = st.slider(
                col,
                float(feature_min[col]), float(feature_max[col]),
                float(feature_mean[col])
            )
        X_in = np.array([values_all[c] for c in X.columns], dtype=float).reshape(1, -1)

    # Choose model to predict with
    model_name = st.selectbox("Model to use", list(models.keys()), index=1)
    model_for_pred = models[model_name]

    # Scale (except for RF if you ever add as a predictor; here all three need scaling)
    X_in_scaled = scaler.transform(X_in)

    pred = model_for_pred.predict(X_in_scaled)[0]
    proba = None
    if hasattr(model_for_pred, "predict_proba"):
        proba = model_for_pred.predict_proba(X_in_scaled)[0][1]  # probability of class 1 (Malignant)

    st.markdown("---")
    st.subheader("Prediction Result")
    if pred == 1:
        if proba is not None:
            st.error(f"Malignant (Cancer) — Confidence: {proba*100:.2f}%")
        else:
            st.error("Malignant (Cancer)")
    else:
        if proba is not None:
            st.success(f"Benign — Confidence: {(1-proba)*100:.2f}%")
        else:
            st.success("Benign")

    with st.expander("Show numeric feature vector used for prediction"):
        show = pd.DataFrame(X_in, columns=X.columns)
        st.dataframe(show, use_container_width=True)
