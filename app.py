
import json
import numpy as np
import pandas as pd
import streamlit as st

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Pipeline AI | Leak Detection",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. MODERN DASHBOARD CSS
# ==========================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #08111f 0%, #101b30 55%, #0b1526 100%);
    color: #e5eefc;
}

[data-testid="stSidebar"] {
    background: #0b1424;
    border-right: 1px solid #263852;
}

[data-testid="stSidebar"] * {
    color: #dbeafe;
}

h1, h2, h3 {
    color: #eaf4ff !important;
    font-weight: 700 !important;
}

p, label, .stMarkdown {
    color: #c4d2e5;
}

.hero {
    background: linear-gradient(110deg, #102b4b, #123d5a, #172554);
    border: 1px solid #23617c;
    border-radius: 18px;
    padding: 28px;
    margin-bottom: 22px;
}

.hero-label {
    color: #67e8f9;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
}

.hero-title {
    color: #ffffff;
    font-size: 32px;
    font-weight: 800;
    margin-top: 10px;
}

.hero-description {
    color: #c5d9ed;
    font-size: 15px;
    margin-top: 10px;
}

.section-card {
    background: #142238;
    border: 1px solid #263c58;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 15px;
}

.section-card h3 {
    color: #67e8f9 !important;
}

.small-label {
    color: #8da9c8;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.status-pill {
    display: inline-block;
    background: #123d37;
    color: #6ee7b7;
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: bold;
    border: 1px solid #176b59;
}

.architecture-box {
    background: #14263d;
    border: 1px solid #28617b;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    margin: 5px 0;
    color: #e0f2fe;
    font-weight: 600;
}

.architecture-arrow {
    text-align: center;
    color: #22d3ee;
    font-size: 24px;
    margin: 2px;
}

div[data-testid="stMetric"] {
    background: #142238;
    border: 1px solid #2a405e;
    padding: 16px;
    border-radius: 13px;
}

div[data-testid="stMetricLabel"] {
    color: #9bb5d3;
}

div[data-testid="stMetricValue"] {
    color: #67e8f9;
}

.stButton button {
    background: linear-gradient(90deg, #0891b2, #2563eb);
    color: white;
    border: none;
    border-radius: 9px;
    font-weight: 600;
}

hr {
    border-color: #263c58;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. LOAD TRAINED MODEL
# ==========================================

@st.cache_data
def load_model():
    with open("demo_model..json", "r") as file:
        return json.load(file)


# ==========================================
# 4. SOFTMAX FUNCTION
# ==========================================

def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


# ==========================================
# 5. GRAPHSAGE-V2 PREDICTION
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

    return softmax(logits)


# ==========================================
# 6. COMMON DASHBOARD HEADER
# ==========================================

st.markdown("""
<div class="hero">
    <div class="hero-label">MULTI-SENSOR INTELLIGENCE SYSTEM</div>
    <div class="hero-title">💧 PIPELINE AI</div>
    <div class="hero-description">
        Deep Learning-Based Detection and Classification of
        Underground Water Pipeline Leaks Using Multi-Sensor Signals
    </div>
    <br>
    <span class="status-pill">● GRAPHSAGE-V2 MODEL</span>
</div>
""", unsafe_allow_html=True)


# ==========================================
# 7. SIDEBAR NAVIGATION
# ==========================================

with st.sidebar:

    st.markdown("## 💧 PIPELINE AI")

    st.caption("Underground Leak Detection System")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Overview",
            "🏗️ System Architecture",
            "📡 Dataset & Sensors",
            "🔍 Leak Prediction",
            "📊 Model Evaluation"
        ]
    )

    st.divider()

    st.markdown("### MODEL STATUS")

    st.markdown(
        '<span class="status-pill">● GraphSAGE-v2</span>',
        unsafe_allow_html=True
    )

    st.caption("Experimental dataset demonstration")

    st.divider()

    st.caption("Final Year Academic Project")


# ==========================================
# 8. LOAD MODEL AND DATA
# ==========================================

try:

    demo = load_model()

    classes = demo["classes"]
    sensors = demo["sensors"]
    samples = demo["samples"]
    levels = demo["levels"]

    st.success("Trained GraphSAGE-v2 model loaded successfully!")

    # ======================================
    # 9. OVERVIEW PAGE
    # ======================================

    if page == "🏠 Overview":

        st.header("System Overview")

        st.write(
            "An experimental multi-sensor classification system "
            "designed to identify underground water pipeline "
            "conditions using a GraphSAGE-v2 model."
        )

        # Main metrics
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Sensor Channels", len(sensors))
        col2.metric("Classification Classes", len(classes))
        col3.metric("Prepared Demo Samples", len(samples))
        col4.metric("Model", "GraphSAGE-v2")

        st.divider()

        # Problem statement
        st.subheader("🎯 Problem Statement")

        st.markdown("""
        Underground water pipeline leaks can be difficult to identify
        because pipelines are located below the ground.

        Different pipeline conditions produce different sensor signal
        patterns. Analysing these signals can help distinguish between
        normal and abnormal pipeline conditions.
        """)

        # Objectives
        st.subheader("🚀 Project Objectives")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("""
            <div class="section-card">
                <h3>01 · Multi-Sensor Analysis</h3>
                Analyse experimental signals collected from
                accelerometers, hydrophones and pressure sensors.
            </div>

            <div class="section-card">
                <h3>02 · Feature Extraction</h3>
                Extract meaningful signal features from
                individual sensor channels.
            </div>
            """, unsafe_allow_html=True)

        with col2:

            st.markdown("""
            <div class="section-card">
                <h3>03 · Graph Learning</h3>
                Apply GraphSAGE-v2 to learn relationships
                between sensor nodes.
            </div>

            <div class="section-card">
                <h3>04 · Classification</h3>
                Classify pipeline conditions into five
                experimental categories.
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        st.subheader("🧠 Proposed Solution")

        st.markdown("""
        The system uses experimental multi-sensor signals,
        feature extraction and graph-based learning to predict
        the pipeline condition.

        The trained GraphSAGE-v2 model analyses the six sensor
        channels and produces probabilities for five classes.
        """)

        st.info(
            "This website demonstrates predictions using prepared "
            "experimental samples. It is not a live pipeline monitoring system."
        )


    # ======================================
    # 10. SYSTEM ARCHITECTURE PAGE
    # ======================================

    elif page == "🏗️ System Architecture":

        st.header("System Architecture")

        st.write(
            "The following workflow represents the processing "
            "pipeline used by the experimental classification system."
        )

        architecture_steps = [
            ("01 · Experimental Sensor Signals",
             "Accelerometers, Hydrophones and Pressure Sensors"),
            ("02 · Signal Preprocessing",
             "Signal reading, windowing and preparation"),
            ("03 · Feature Extraction",
             "16 features extracted for each sensor channel"),
            ("04 · Sensor Graph Construction",
             "Six sensor channels represented as graph nodes"),
            ("05 · GraphSAGE-v2 Model",
             "Feature encoding and graph message passing"),
            ("06 · Classification Output",
             "Five pipeline condition classes with probabilities")
        ]

        for title, description in architecture_steps:

            st.markdown(
                f"""
                <div class="architecture-box">
                    <div style="color:#67e8f9;font-size:16px;">
                        {title}
                    </div>
                    <div style="color:#b9cce2;font-size:13px;margin-top:6px;">
                        {description}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if title != architecture_steps[-1][0]:

                st.markdown(
                    '<div class="architecture-arrow">↓</div>',
                    unsafe_allow_html=True
                )

        st.divider()

        st.subheader("GraphSAGE-v2 Processing")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("""
            <div class="section-card">
                <h3>Sensor Feature Encoder</h3>
                Each sensor node has 16 extracted features.
                The encoder converts the sensor features
                into a learned representation.
            </div>
            """, unsafe_allow_html=True)

        with col2:

            st.markdown("""
            <div class="section-card">
                <h3>Graph Message Passing</h3>
                The model learns sensor relationships and
                combines node information through graph
                message-passing layers.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="section-card">
            <h3>Final Classification</h3>
            The learned representations are passed to the
            classification layer, which produces probabilities
            for the five pipeline condition classes.
        </div>
        """, unsafe_allow_html=True)

        st.caption(
            "The website uses the exported NumPy inference implementation "
            "of the trained GraphSAGE-v2 model."
        )


    # ======================================
    # 11. DATASET AND SENSORS PAGE
    # ======================================

    elif page == "📡 Dataset & Sensors":

        st.header("Dataset & Sensor Information")

        st.write(
            "The project uses an experimental underground "
            "water pipeline dataset."
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Recordings", "40")
        col2.metric("Windows", "2,360")
        col3.metric("Sensors", "6")
        col4.metric("Classes", "5")

        st.divider()

        st.subheader("Sensor Channels")

        sensor_info = pd.DataFrame({
            "Sensor Channel": ["A1", "A2", "H1", "H2", "P1", "P2"],
            "Sensor Type": [
                "Accelerometer",
                "Accelerometer",
                "Hydrophone",
                "Hydrophone",
                "Dynamic Pressure",
                "Dynamic Pressure"
            ],
            "Purpose": [
                "Measures pipeline vibration",
                "Measures pipeline vibration",
                "Measures acoustic signals",
                "Measures acoustic signals",
                "Measures pressure variation",
                "Measures pressure variation"
            ]
        })

        st.dataframe(
            sensor_info,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        st.subheader("Classification Categories")

        class_descriptions = {
            "CC": "Class CC - Experimental pipeline condition.",
            "GL": "Class GL - Experimental pipeline condition.",
            "LC": "Class LC - Experimental pipeline condition.",
            "NL": "Class NL - Experimental pipeline condition.",
            "OL": "Class OL - Experimental pipeline condition."
        }

        for class_name in classes:

            st.markdown(
                f"""
                <div class="section-card">
                    <b style="color:#67e8f9;">{class_name}</b>
                    <div style="color:#c4d2e5;">
                        {class_descriptions.get(class_name, "Experimental pipeline class.")}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.divider()

        st.subheader("Dataset Structure")

        dataset_info = pd.DataFrame({
            "Parameter": [
                "Experimental recordings",
                "Pipeline configurations",
                "Classification classes",
                "Flow conditions",
                "Total signal windows",
                "Window duration",
                "Window hop"
            ],
            "Value": [
                "40",
                "2 (BR and LO)",
                "5",
                "4",
                "2,360",
                "1 second",
                "0.5 second"
            ]
        })

        st.dataframe(
            dataset_info,
            use_container_width=True,
            hide_index=True
        )

        st.info(
            "The website's prediction model uses 16 features "
            "for each of the six sensor nodes (96 node features per sample)."
        )


    # ======================================
    # 12. LEAK PREDICTION PAGE
    # ======================================

    elif page == "🔍 Leak Prediction":

        st.header("Interactive Leak Classification")

        st.write(
            "Select a prepared experimental sample and signal "
            "noise level to view the GraphSAGE-v2 prediction."
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

        st.divider()

        st.subheader("Prediction Results")

        col1, col2, col3 = st.columns(3)

        col1.metric("Predicted Class", predicted_class)
        col2.metric("Confidence", f"{confidence:.2f}%")
        col3.metric("Actual Class", actual_class)

        if predicted_class == actual_class:

            st.success(
                "Correct prediction! The predicted class matches "
                "the actual experimental label."
            )

        else:

            st.error(
                "Incorrect prediction. The predicted class differs "
                "from the actual experimental label."
            )

        st.markdown(
            f"**Sample number:** {sample_number}  \n"
            f"**Network configuration:** {selected_sample['net']}  \n"
            f"**Signal noise level:** {noise_level}"
        )

        st.divider()

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

        st.caption(
            "Confidence represents the highest predicted class probability. "
            "It is not a guarantee that the prediction is correct."
        )


    # ======================================
    # 13. MODEL EVALUATION PAGE
    # ======================================

    elif page == "📊 Model Evaluation":

        st.header("Model Evaluation Dashboard")

        st.write(
            "Evaluation of all prepared demonstration samples "
            "using their clean signal features."
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

        # Confusion matrix
        cm = np.zeros(
            (len(classes), len(classes)),
            dtype=int
        )

        for actual, predicted in zip(actuals, predictions):

            cm[actual, predicted] += 1

        # Overall metrics
        correct = int(np.sum(actuals == predictions))

        total = len(actuals)

        incorrect = total - correct

        accuracy = (correct / total) * 100 if total else 0

        col1, col2, col3 = st.columns(3)

        col1.metric("Demo Accuracy", f"{accuracy:.2f}%")
        col2.metric("Correct Predictions", f"{correct}/{total}")
        col3.metric("Incorrect Predictions", incorrect)

        st.divider()

        # Confusion matrix
        st.subheader("Confusion Matrix")

        st.caption(
            "Rows represent actual classes. Columns represent predicted classes."
        )

        cm_df = pd.DataFrame(
            cm,
            index=classes,
            columns=classes
        )

        st.table(cm_df)

        # Class-wise metrics
        st.divider()

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

        st.subheader("Performance Comparison")

        chart_df = metrics_df.set_index("Class")[
            [
                "Precision (%)",
                "Recall (%)",
                "F1-score (%)"
            ]
        ]

        st.bar_chart(chart_df)

        # Incorrect predictions
        st.divider()

        st.subheader("Misclassification Analysis")

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

        st.info(
            f"""
            Evaluation scope: {total} prepared demonstration samples.

            Demo accuracy: {accuracy:.2f}%.

            This result is separate from the full 590-window
            held-out test evaluation. It should not be interpreted
            as the model's full test-set accuracy.
            """
        )


    # ======================================
    # 14. COMMON PROJECT SCOPE
    # ======================================

    st.divider()

    with st.expander("ℹ️ Project Scope and Limitations"):

        st.markdown("""
        - This demonstration uses prepared experimental dataset samples.
        - The website does not currently process live sensor signals.
        - It does not provide real-time pipeline monitoring.
        - Physical sensor installation, live streaming and edge deployment
          are outside the scope of this demonstration.
        - Model performance on experimental samples may differ from
          performance in real underground pipeline environments.
        """)

    st.caption(
        "PIPELINE AI | Academic Final Year Project | GraphSAGE-v2"
    )


except Exception as e:

    st.error(
        f"An error occurred while loading or running the application: {e}"
    )

    st.info(
        "Please verify that demo_model..json exists in your "
        "GitHub repository and that its filename matches "
        "the filename used in load_model()."
    )
