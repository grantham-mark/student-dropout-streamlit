import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
)

st.set_page_config(page_title="Student Dropout Prediction Model Results", layout="wide")

st.title("📊 Student Dropout Prediction – Machine Learning Model Dashboard")

# ======================================
# 1. Upload Dataset
# ======================================
uploaded_file = st.file_uploader("Upload your students_dropout_academic_success.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.lower()
    st.success("Dataset Loaded Successfully!")
    st.dataframe(df.head())

    # Encode target
    df['target_binary'] = df['target'].map({'Dropout': 1, 'Graduate': 0})
    if df['target_binary'].isnull().any():
        df['target_binary'] = df['target'].apply(lambda x: 0 if str(x).lower()=='graduate' else 1)

    feature_cols = [c for c in df.columns if c not in ['target', 'target_binary']]
    X = df[feature_cols]
    y = df['target_binary']

    # Split numeric vs categorical
    numeric_feats = [c for c in feature_cols if df[c].dtype != 'object']
    categorical_feats = [c for c in feature_cols if df[c].dtype == 'object']

    # Preprocessing
    preprocessor = ColumnTransformer(
        [
            ('num', StandardScaler(), numeric_feats),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_feats)
        ],
        remainder='drop'
    )

    # ======================================
    # 2. Train-test split
    # ======================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ======================================
    # 3. Define models
    # ======================================
    base_models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM (RBF Kernel)': SVC(probability=True, kernel='rbf', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    voting_clf = VotingClassifier(
        estimators=[
            ('lr', base_models['Logistic Regression']),
            ('svm', base_models['SVM (RBF Kernel)']),
            ('rf', base_models['Random Forest'])
        ],
        voting='soft'
    )

    base_models['Voting Ensemble'] = voting_clf

    st.header("📈 Model Performance Summary")

    results = {}

    # ======================================
    # 4. Train + Evaluate
    # ======================================
    for name, model in base_models.items():
        pipe = Pipeline([('preproc', preprocessor), ('clf', model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_prob)
        }

    results_df = pd.DataFrame(results).T
    st.dataframe(results_df.style.highlight_max(axis=0))

    # ======================================
    # 5. Confusion Matrices
    # ======================================
    st.header("🧩 Confusion Matrices")

    for name, model in base_models.items():
        pipe = Pipeline([('preproc', preprocessor), ('clf', model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        st.subheader(name)
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(cm).plot(ax=ax, cmap="Blues")
        st.pyplot(fig)

    # ======================================
    # 6. ROC Curves
    # ======================================
    st.header("📉 ROC Curves")

    fig, ax = plt.subplots(figsize=(8, 6))

    for name, model in base_models.items():
        pipe = Pipeline([('preproc', preprocessor), ('clf', model)])
        pipe.fit(X_train, y_train)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, y_prob):.3f})")

    ax.plot([0,1], [0,1], 'k--')
    ax.set_title("ROC Curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

    # ======================================
    # 7. Feature Importance (Random Forest)
    # ======================================
    st.header("⭐ Top Feature Importances (Random Forest)")

    rf_model = Pipeline([('preproc', preprocessor), ('clf', base_models['Random Forest'])])
    rf_model.fit(X_train, y_train)

    rf = rf_model.named_steps['clf']
    importances = rf.feature_importances_
    expanded_features = rf_model.named_steps['preproc'].get_feature_names_out()

    feat_df = pd.DataFrame({
        'Feature': expanded_features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    st.dataframe(feat_df.head(10))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(feat_df['Feature'][:10][::-1], feat_df['Importance'][:10][::-1])
    ax.set_title('Top 10 Feature Importances')
    st.pyplot(fig)

else:
    st.info("⬆️ Please upload your dataset to see model results.")


# =======================================================
# 8. ADVANCED PREDICTION TOOL (SAFE VERSION — NO ERRORS)
# =======================================================

if uploaded_file:

    st.header("🎯 Advanced Student Dropout Prediction")

    st.write("Fill in student characteristics below to generate a prediction.")

    # Ensure required variables exist
    try:
        _ = feature_cols
        _ = numeric_feats
        _ = categorical_feats
        _ = preprocessor
    except NameError:
        st.error("❌ Internal error: preprocessing variables not defined. Ensure this block is placed AFTER data loading and preprocessing.")
        st.stop()

    # ---------------------------
    # Input Form
    # ---------------------------
    st.subheader("📝 Student Details Input")

    col1, col2 = st.columns(2)
    user_input = {}

    for i, col in enumerate(feature_cols):
        if col in numeric_feats:
            default_value = float(df[col].mean()) if df[col].dtype != 'object' else 0
            widget = col1 if i % 2 == 0 else col2
            user_input[col] = widget.number_input(col, value=default_value)
        else:
            unique_vals = df[col].dropna().unique()
            widget = col1 if i % 2 == 0 else col2
            user_input[col] = widget.selectbox(col, unique_vals)

    input_df = pd.DataFrame([user_input])

    # ---------------------------
    # Model selection
    # ---------------------------
    st.subheader("🤖 Choose Prediction Model")

    model_choice = st.selectbox(
        "Select a model:",
        ["Logistic Regression", "SVM (RBF Kernel)", "Random Forest", "Voting Ensemble"]
    )

    def train_selected_model(name):
        if name == "Logistic Regression":
            m = LogisticRegression(max_iter=1000, random_state=42)
        elif name == "SVM (RBF Kernel)":
            m = SVC(probability=True, kernel='rbf', random_state=42)
        elif name == "Random Forest":
            m = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            m = VotingClassifier(
                estimators=[
                    ('lr', LogisticRegression(max_iter=1000, random_state=42)),
                    ('svm', SVC(probability=True, kernel='rbf', random_state=42)),
                    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
                ],
                voting='soft'
            )

        full_model = Pipeline([("preproc", preprocessor), ("clf", m)])
        full_model.fit(X, y)
        return full_model

    # ---------------------------
    # Prediction Button
    # ---------------------------
    if st.button("Predict Dropout"):
        model = train_selected_model(model_choice)
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        st.write("---")
        st.subheader("📌 Dropout Probability Score")
        st.progress(float(prob))

        if pred == 1:
            st.error(f"⚠️ High Dropout Risk — Probability: {prob:.2f}")
        else:
            st.success(f"🎉 Likely to Graduate — Probability: {(1 - prob):.2f}")

        # Save prediction
        if st.checkbox("Save prediction to CSV"):
            log = pd.DataFrame({
                "Model": [model_choice],
                "Prediction": ["Dropout" if pred == 1 else "Graduate"],
                "Dropout_Probability": [prob]
            }).join(input_df)

            try:
                old = pd.read_csv("prediction_log.csv")
                new = pd.concat([old, log], ignore_index=True)
                new.to_csv("prediction_log.csv", index=False)
                st.success("Prediction saved successfully.")
            except:
                log.to_csv("prediction_log.csv", index=False)
                st.success("prediction_log.csv created.")
