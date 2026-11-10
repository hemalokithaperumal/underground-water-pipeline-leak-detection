
import json
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Underground Water Pipeline Leak Detection",
    page_icon="💧",
    layout="wide"
)


# ==========================================
# 1. LOAD TRAINED MODEL
# ==========================================

@st.cache_data
def load_model():
    with open("demo_model..json", "r") as file:
        return json.load(file)


# ==========================================
# 2. SOFTMAX FUNCTION
# ==========================================

def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return np.exp(x) / np.sum(np.exp(x))


# ==========================================
# 3. GRAPHSAGE-V2 PREDICTION
# ==========================================

def predict(features, demo):

    model = demo["model"]
    scaler = demo["scaler"]

    x = np.asarray(features, dtype=np.float32)

    mu = np.asarray(scaler["mu"], dtype=np.float32)
    sd = np.asarray(scaler["sd"], dtype=np.float32)

    x = (x - mu) / np.maximum(sd, 1e-8)

    # Sensor feature encoder
    h = np.einsum(
        "nf,nfh->nh",
        x,
        np.asarray(model["enc_W"])
    ) + np.asarray(model["enc_b"])

    h = np.maximum(h, 0)

    # Learned sensor graph
    L = np.asarray(model["L"], dtype=np.float32).copy()

    np.fill_diagonal(L, -np.inf)

    A = np.exp(L - np.max(L, axis=1, keepdims=True))
    A = A / A.sum(axis=1, keepdims=True)

    # GraphSAGE layer 1
    neighbor = A @ h

    h = (
        h @ np.asarray(model["ls1_W"]).T
        + np.asarray(model["ls1_b"])
        + neighbor @ np.asarray(model["ln1_W"]).T
        + np.asarray(model["ln1_b"])
    )

    h = np.maximum(h, 0)

    # GraphSAGE layer 2
    neighbor = A @ h

    h = (
        h @ np.asarray(model["ls2_W"]).T
        + np.asarray(model["ls2_b"])
        + neighbor @ np.asarray(model["ln2_W"]).T
        + np.asarray(model["ln2_b"])
    )

    h = np.maximum(h, 0)

    # Classification layer
    h = h.reshape(1, -1)

    logits = (
        h @ np.asarray(model["out_W"]).T
        + np.asarray(model["out_b"])
    )[0]

    probabilities = softmax(logits)

    return probabilities


# ==========================================
# 4. WEBSITE HEADER
# ==========================================

st.title("💧 Underground Water Pipeline Leak Detection")

st.write(
    "A GraphSAGE-v2 based detection and classification "
    "system using multi-sensor signals."
)

st.markdown(
    """
    This project demonstrates the classification of
    underground water pipeline conditions using
    experimental multi-sensor data.
    """
)

st.divider()


# ==========================================
# 5. MAIN APPLICATION
# ==========================================

try:

    demo = load_model()

    classes = demo["classes"]
    sensors = demo["sensors"]
    samples = demo["samples"]
    levels = demo["levels"]

    st.success(
        "Trained GraphSAGE-v2 model loaded successfully!"
    )

    # ======================================
    # MODEL INFORMATION
    # ======================================

    st.header("🧠 Model Information")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Classification Categories",
        len(classes)
    )

    col2.metric(
        "Sensor Channels",
        len(sensors)
    )

    col3.metric(
        "Prepared Test Samples",
        len(samples)
    )

    st.write("**Sensor channels:**", ", ".join(sensors))
    st.write("**Classification classes:**", ", ".join(classes))

    st.divider()

    # ======================================
    # 6. SAMPLE PREDICTION
    # ======================================

    st.header("🔍 Leak Classification Demo")

    st.write(
        "Select a prepared experimental sample and noise level "
        "to view the model prediction."
    )

    col1, col2 = st.columns(2)

    with col1:

        sample_number = st.selectbox(
            "Select Test Sample",
            options=list(range(1, len(samples) + 1)),
            index=0
        )

    with col2:

        noise_level = st.selectbox(
            "Select Signal Noise Level",
            options=levels,
            index=0
        )

    selected_sample = samples[sample_number - 1]

    features = selected_sample["x"][noise_level]

    probabilities = predict(features, demo)

    predicted_index = int(np.argmax(probabilities))

    predicted_class = classes[predicted_index]

    actual_class = classes[int(selected_sample["label"])]

    confidence = float(probabilities[predicted_index]) * 100

    # Prediction results
    st.subheader("Prediction Result")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Predicted Class",
        predicted_class
    )

    col2.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )

    col3.metric(
        "Actual Class",
        actual_class
    )

    if predicted_class == actual_class:

        st.success(
            "Correct prediction! The predicted class matches "
            "the actual class."
        )

    else:

        st.error(
            "Incorrect prediction. The predicted class differs "
            "from the actual class."
        )

    st.write(
        "**Network configuration:**",
        selected_sample["net"]
    )

    # ======================================
    # CLASS PROBABILITY CHART
    # ======================================

    st.subheader("Class Probability Distribution")

    probability_df = pd.DataFrame({
        "Class": classes,
        "Probability (%)": probabilities * 100
    })

    st.bar_chart(
        probability_df.set_index("Class")
    )

    st.dataframe(
        probability_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ======================================
    # 7. AUTOMATED MODEL EVALUATION
    # ======================================

    st.header("📊 Model Evaluation Dashboard")

    st.write(
        "The following evaluation uses all prepared demonstration "
        "samples with clean signals."
    )

    actuals = []
    predictions = []

    for sample in samples:

        features = sample["x"]["clean"]

        probabilities = predict(features, demo)

        predicted_index = int(np.argmax(probabilities))

        actuals.append(int(sample["label"]))

        predictions.append(predicted_index)

    actuals = np.array(actuals)

    predictions = np.array(predictions)

    # ======================================
    # CONFUSION MATRIX CALCULATION
    # ======================================

    cm = np.zeros(
        (len(classes), len(classes)),
        dtype=int
    )

    for actual, predicted in zip(actuals, predictions):

        cm[actual, predicted] += 1

    # ======================================
    # OVERALL PERFORMANCE
    # ======================================

    correct = int(np.sum(actuals == predictions))

    total = len(actuals)

    incorrect = total - correct

    accuracy = (correct / total) * 100 if total else 0

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Overall Accuracy",
        f"{accuracy:.2f}%"
    )

    col2.metric(
        "Correct Predictions",
        f"{correct}/{total}"
    )

    col3.metric(
        "Incorrect Predictions",
        incorrect
    )

    # ======================================
    # CONFUSION MATRIX HEATMAP
    # ======================================

    st.subheader("Confusion Matrix Heatmap")

    st.write(
        "Rows represent actual classes; columns represent "
        "predicted classes."
    )

    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual {c}" for c in classes],
        columns=[f"Predicted {c}" for c in classes]
    )

    st.dataframe(
        cm_df.style.background_gradient(
            cmap="Blues"
        ).format("{:.0f}"),
        use_container_width=True
    )

    # ======================================
    # CLASS-WISE PERFORMANCE
    # ======================================

    st.subheader("Class-wise Performance")

    class_metrics = []

    for i, class_name in enumerate(classes):

        tp = cm[i, i]

        predicted_count = cm[:, i].sum()

        actual_count = cm[i, :].sum()

        precision = (
            tp / predicted_count
            if predicted_count > 0 else 0
        )

        recall = (
            tp / actual_count
            if actual_count > 0 else 0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0 else 0
        )

        class_metrics.append({

            "Class": class_name,

            "Precision (%)": round(precision * 100, 2),

            "Recall (%)": round(recall * 100, 2),

            "F1-score (%)": round(f1 * 100, 2)

        })

    metrics_df = pd.DataFrame(class_metrics)

    st.dataframe(
        metrics_df,
        use_container_width=True,
        hide_index=True
    )

    # ======================================
    # CLASS-WISE PERFORMANCE CHART
    # ======================================

    st.subheader("Performance Comparison")

    chart_df = metrics_df.set_index("Class")[
        ["Precision (%)", "Recall (%)", "F1-score (%)"]
    ]

    st.bar_chart(chart_df)

    # ======================================
    # MISCLASSIFICATION DETAILS
    # ======================================

    st.subheader("Incorrect Predictions Analysis")

    wrong_indices = np.where(actuals != predictions)[0]

    if len(wrong_indices) > 0:

        wrong_data = []

        for index in wrong_indices:

            wrong_data.append({

                "Sample Number": int(index + 1),

                "Actual Class": classes[int(actuals[index])],

                "Predicted Class": classes[int(predictions[index])]

            })

        wrong_df = pd.DataFrame(wrong_data)

        st.dataframe(
            wrong_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.success(
            "All demonstration samples were classified correctly."
        )

    st.caption(
        "Note: This dashboard evaluates 60 prepared demonstration "
        "samples. It is not the full 590-window held-out test result."
    )

    st.divider()

    # ======================================
    # 8. PROJECT LIMITATIONS
    # ======================================

    st.header("ℹ️ Project Scope and Limitations")

    st.info(
        """
        This demonstration uses prepared test samples from the
        experimental dataset.

        It does not currently process live sensor data or provide
        real-time pipeline monitoring.

        Physical sensor installation, live streaming and
        edge deployment are outside the scope of this demonstration.
        """
    )

    st.caption(
        "Developed as an academic final-year project."
    )


except Exception as e:

    st.error(
        f"An error occurred while loading or running the application: {e}"
    )

    st.info(
        "Please verify that the model JSON file is present in "
        "the GitHub repository and that its filename matches "
        "the filename used in load_model()."
    )
