
import json
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Pipeline Leak Detection",
    page_icon="💧",
    layout="wide"
)


# -----------------------------------
# Load trained model
# -----------------------------------

@st.cache_data
def load_model():
    with open("demo_model..json", "r") as file:
        return json.load(file)


# -----------------------------------
# Softmax function
# -----------------------------------

def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


# -----------------------------------
# GraphSAGE-v2 prediction
# -----------------------------------

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


# -----------------------------------
# Main website
# -----------------------------------

st.title("💧 Underground Water Pipeline Leak Detection")

st.write(
    "GraphSAGE-v2 based detection and classification "
    "using multi-sensor signals."
)

st.divider()

try:
    demo = load_model()

    classes = demo["classes"]
    sensors = demo["sensors"]
    samples = demo["samples"]
    levels = demo["levels"]

    st.success("Trained GraphSAGE-v2 model loaded successfully!")

    st.subheader("Model Information")

    col1, col2, col3 = st.columns(3)

    col1.metric("Classification Categories", len(classes))
    col2.metric("Sensors", len(sensors))
    col3.metric("Prepared Test Samples", len(samples))

    st.write("**Sensor channels:**", ", ".join(sensors))
    st.write("**Classes:**", ", ".join(classes))
    
    # -----------------------------------
    # Automated Model Evaluation
    # -----------------------------------

    st.divider()
    st.header("📊 Model Evaluation Dashboard")

    @st.cache_data
    def evaluate_model(demo):
        actuals = []
        predictions = []

        for sample in demo["samples"]:
            features = sample["x"]["clean"]

            probabilities = predict(features, demo)
            predicted_index = int(np.argmax(probabilities))

            actuals.append(int(sample["label"]))
            predictions.append(predicted_index)

        actuals = np.array(actuals)
        predictions = np.array(predictions)

        cm = np.zeros((len(classes), len(classes)), dtype=int)

        for actual, predicted in zip(actuals, predictions):
            cm[actual, predicted] += 1

        correct = int(np.sum(actuals == predictions))
        total = len(actuals)
        accuracy = correct / total * 100

        return actuals, predictions, cm, correct, total, accuracy

    actuals, predictions, cm, correct, total, accuracy = (
        evaluate_model(demo)
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Overall Accuracy", f"{accuracy:.2f}%")
    col2.metric("Correct Predictions", f"{correct}/{total}")
    col3.metric("Incorrect Predictions", total - correct)

    st.subheader("Confusion Matrix")

    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual {c}" for c in classes],
        columns=[f"Predicted {c}" for c in classes]
    )

    st.dataframe(cm_df, use_container_width=True)

    st.caption(
        "Rows represent actual classes; columns represent predicted classes."
    )

    st.subheader("Class-wise Performance")

    rows = []

    for i, class_name in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0
        )

        rows.append({
            "Class": class_name,
            "Precision (%)": round(precision * 100, 2),
            "Recall (%)": round(recall * 100, 2),
            "F1-score (%)": round(f1 * 100, 2)
        })

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True
    )


    st.divider()

    st.subheader("Leak Classification Demo")

    sample_index = st.selectbox(
        "Select a prepared test sample",
        range(len(samples)),
        format_func=lambda i: f"Sample {i + 1}"
    )

    sample = samples[sample_index]

    noise_level = st.selectbox(
        "Select signal noise level",
        levels,
        format_func=lambda level:
            "Clean signal" if level == "clean"
            else f"{level} dB noise"
    )

    features = sample["x"][noise_level]

    probabilities = predict(features, demo)

    predicted_index = int(np.argmax(probabilities))
    predicted_class = classes[predicted_index]

    actual_class = classes[int(sample["label"])]

    confidence = float(probabilities[predicted_index]) * 100

    st.divider()

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    col1.metric("Predicted Class", predicted_class)
    col2.metric("Prediction Confidence", f"{confidence:.2f}%")

    if predicted_class == actual_class:
        st.success("Prediction matches the actual class.")
    else:
        st.warning(
            f"Prediction differs from actual class: {actual_class}"
        )

    st.subheader("Class Probability")

    chart_data = pd.DataFrame(
        {"Probability (%)": probabilities * 100},
        index=classes
    )

    st.bar_chart(chart_data)

    st.write("**Network configuration:**", sample["net"])

    st.caption(
        "This demonstration uses prepared test samples from the "
        "experimental dataset. It does not process live sensor data."
    )

except Exception as error:
    st.error("Unable to load or run the prediction model.")
    st.exception(error)
