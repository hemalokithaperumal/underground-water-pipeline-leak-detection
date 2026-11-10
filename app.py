import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Pipeline AI | Leak Detection",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# 2. DASHBOARD CSS
# =====================================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #08111f, #101b30, #0b1526);
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

p, label {
    color: #c4d2e5;
}

/* Compact Overview heading */
.compact-header {
    background: linear-gradient(120deg, #102b4b, #123d5a);
    border: 1px solid #23617c;
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 22px;
}

.compact-title {
    color: #67e8f9;
    font-size: 28px;
    font-weight: 800;
    display: flex;
    align-items: center;
    gap: 14px;
    flex-wrap: wrap;
}

.compact-description {
    color: #c5d9ed;
    font-size: 14px;
    margin-top: 8px;
    line-height: 1.6;
}

.status-pill {
    display: inline-block;
    background: #123d37;
    color: #6ee7b7;
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: bold;
    border: 1px solid #176b59;
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


# =====================================================
# 3. LOAD EXPORTED MODEL
# =====================================================

@st.cache_data
def load_model():
    with open("demo_model..json", "r") as file:
        return json.load(file)


# =====================================================
# 4. SOFTMAX
# =====================================================

def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


# =====================================================
# 5. GRAPHSAGE-V2 NUMPY INFERENCE
# =====================================================

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


# =====================================================
# 6. CHART STYLING
# =====================================================

def style_chart(fig, height=400):

    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dbeafe", family="Arial"),
        margin=dict(l=20, r=20, t=55, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    fig.update_xaxes(
        gridcolor="#263c58",
        zerolinecolor="#263c58"
    )

    fig.update_yaxes(
        gridcolor="#263c58",
        zerolinecolor="#263c58"
    )

    return fig


# =====================================================
# 7. SIDEBAR NAVIGATION
# =====================================================

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
            "📊 Model Evaluation",
            "👥 About Project"
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


# =====================================================
# 8. LOAD DATA AND DISPLAY SELECTED PAGE
# =====================================================

try:

    demo = load_model()

    classes = demo["classes"]
    sensors = demo["sensors"]
    samples = demo["samples"]
    levels = demo["levels"]

    st.success("Trained GraphSAGE-v2 model loaded successfully!")

    # -------------------------------------------------
    # COMPACT HEADER: OVERVIEW ONLY
    # -------------------------------------------------

    if page == "🏠 Overview":

        st.markdown("""
        <div class="compact-header">
            <div class="compact-title">
                💧 PIPELINE AI
                <span class="status-pill">● GRAPHSAGE-V2</span>
            </div>
            <div class="compact-description">
                Deep Learning-Based Detection and Classification
                of Underground Water Pipeline Leaks Using Multi-Sensor Signals
            </div>
        </div>
        """, unsafe_allow_html=True)

    # =================================================
    # PAGE 1: OVERVIEW
    # =================================================

    if page == "🏠 Overview":

        st.header("System Overview")

        st.write(
            "An experimental multi-sensor classification system "
            "designed to identify underground water pipeline "
            "conditions using a GraphSAGE-v2 model."
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Sensor Channels", len(sensors))
        col2.metric("Classification Classes", len(classes))
        col3.metric("Prepared Demo Samples", len(samples))
        col4.metric("Model", "GraphSAGE-v2")

        st.divider()

        st.subheader("🎯 Problem Statement")

        st.markdown("""
        Underground water pipeline leaks can be difficult to identify
        because pipelines are located below the ground.

        Different pipeline conditions produce different sensor signal
        patterns. Analysing these signals can help distinguish between
        normal and abnormal pipeline conditions.
        """)

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

        The trained GraphSAGE-v2 model analyses six sensor
        channels and produces probabilities for five classes.
        """)

        st.info(
            "This website demonstrates predictions using prepared "
            "experimental samples. It is not a live pipeline monitoring system."
        )

    # =================================================
    # PAGE 2: SYSTEM ARCHITECTURE
    # =================================================

    elif page == "🏗️ System Architecture":

        st.header("System Architecture")

        st.write(
            "The following workflow represents the processing "
            "pipeline used by the experimental classification system."
        )

        architecture_steps = [
            (
                "01 · Experimental Sensor Signals",
                "Accelerometers, Hydrophones and Pressure Sensors"
            ),
            (
                "02 · Signal Preprocessing",
                "Signal reading, windowing and preparation"
            ),
            (
                "03 · Feature Extraction",
                "16 features extracted for each sensor channel"
            ),
            (
                "04 · Sensor Graph Construction",
                "Six sensor channels represented as graph nodes"
            ),
            (
                "05 · GraphSAGE-v2 Model",
                "Feature encoding and graph message passing"
            ),
            (
                "06 · Classification Output",
                "Five pipeline condition classes with probabilities"
            )
        ]

        for i, (title, description) in enumerate(architecture_steps):

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

            if i < len(architecture_steps) - 1:

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
                The encoder converts sensor features into
                a learned representation.
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
            for five pipeline condition classes.
        </div>
        """, unsafe_allow_html=True)

        st.caption(
            "The website uses the exported NumPy inference "
            "implementation of the trained GraphSAGE-v2 model."
        )

    # =================================================
    # PAGE 3: DATASET & SENSORS
    # =================================================

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

        st.caption(
            "These are the experimental dataset class codes. "
            "Their detailed meanings should follow the dataset documentation."
        )

        class_cols = st.columns(len(classes))

        for i, class_name in enumerate(classes):

            with class_cols[i]:

                st.markdown(
                    f"""
                    <div class="section-card" style="text-align:center;">
                        <h3>{class_name}</h3>
                        <p>Dataset class label</p>
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
            "The deployed prediction model uses 16 features "
            "for each of six sensor nodes (96 node features per sample)."
        )

    # =================================================
    # PAGE 4: LEAK PREDICTION
    # =================================================

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
            f"""
            **Sample number:** {sample_number}

            **Network configuration:** {selected_sample['net']}

            **Signal noise level:** {noise_level}
            """
        )

        st.divider()

        st.subheader("Class Probability Distribution")

        probability_df = pd.DataFrame({
            "Class": classes,
            "Probability (%)": probabilities * 100
        })

        probability_fig = px.bar(
            probability_df,
            x="Class",
            y="Probability (%)",
            text=probability_df["Probability (%)"].map(
                lambda value: f"{value:.2f}%"
            ),
            color="Probability (%)",
            color_continuous_scale=[
                "#164e63",
                "#0891b2",
                "#2563eb",
                "#67e8f9"
            ]
        )

        probability_fig.update_traces(
            textposition="outside",
            cliponaxis=False
        )

        probability_fig.update_yaxes(
            range=[
                0,
                max(105, float(probabilities.max() * 100) + 10)
            ],
            title="Probability (%)"
        )

        probability_fig.update_layout(
            coloraxis_showscale=False
        )

        probability_fig = style_chart(probability_fig, 420)

        st.plotly_chart(
            probability_fig,
            use_container_width=True
        )

        st.dataframe(
            probability_df.round(2),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Confidence represents the highest predicted class probability. "
            "It is not a guarantee that the prediction is correct."
        )

    # =================================================
    # PAGE 5: MODEL EVALUATION
    # =================================================

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

        # Overall performance
        correct = int(np.sum(actuals == predictions))
        total = len(actuals)
        incorrect = total - correct

        accuracy = correct / total * 100 if total else 0

        col1, col2, col3 = st.columns(3)

        col1.metric("Demo Accuracy", f"{accuracy:.2f}%")
        col2.metric("Correct Predictions", f"{correct}/{total}")
        col3.metric("Incorrect Predictions", incorrect)

        st.caption(
            "These metrics are calculated on the prepared demonstration samples."
        )

        st.divider()

        # Confusion matrix
        st.subheader("Confusion Matrix")

        st.caption(
            "Rows represent actual classes. "
            "Columns represent predicted classes."
        )

        cm_df = pd.DataFrame(
            cm,
            index=classes,
            columns=classes
        )

        st.table(cm_df)

        st.divider()

        # Class-wise metrics
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

        st.divider()

        # Grouped performance chart
        st.subheader("Precision vs Recall vs F1-score")

        chart_df = metrics_df.melt(
            id_vars="Class",
            value_vars=[
                "Precision (%)",
                "Recall (%)",
                "F1-score (%)"
            ],
            var_name="Metric",
            value_name="Score"
        )

        performance_fig = px.bar(
            chart_df,
            x="Class",
            y="Score",
            color="Metric",
            barmode="group",
            text="Score",
            color_discrete_map={
                "Precision (%)": "#22d3ee",
                "Recall (%)": "#3b82f6",
                "F1-score (%)": "#a78bfa"
            }
        )

        performance_fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside",
            cliponaxis=False
        )

        performance_fig.update_yaxes(
            title="Performance (%)",
            range=[0, 115]
        )

        performance_fig = style_chart(performance_fig, 450)

        st.plotly_chart(
            performance_fig,
            use_container_width=True
        )

        st.divider()

        # Misclassification analysis
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

            st.warning(
                f"{len(wrong_indices)} demonstration samples "
                "were misclassified."
            )

        else:

            st.success(
                "All demonstration samples were classified correctly."
            )

        st.info(
            f"""
            **Evaluation scope:** {total} prepared demonstration samples.

            **Demo accuracy:** {accuracy:.2f}%.

            This result is separate from the full 590-window
            held-out test evaluation. It should not be interpreted
            as the model's full test-set accuracy.
            """
        )

    # =================================================
    # PAGE 6: ABOUT PROJECT
    # =================================================

    elif page == "👥 About Project":

        st.header("About the Project")

        st.markdown("""
        <div class="section-card">
            <h3>💧 Project Title</h3>
            Deep Learning-Based Detection and Classification
            of Underground Water Pipeline Leaks Using
            Multi-Sensor Signals
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Institution")

        st.markdown("""
        **Saveetha Engineering College**

        Department of Computer Science and Engineering
        (Internet of Things)

        Chennai, Tamil Nadu
        """)

        st.divider()

        st.subheader("Project Team")

        team_df = pd.DataFrame({
            "Role": [
                "Project Student",
                "Project Student"
            ],
            "Name": [
                "Hema Lokitha P",
                "Gayathiri Roshini M"
            ]
        })

        st.dataframe(
            team_df,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Project Guide")

        st.markdown("""
        **Mr. K. Premnath**

        Department of Computer Science and Engineering
        """)

        st.divider()

        st.subheader("Technology Stack")

        technology_df = pd.DataFrame({
            "Technology": [
                "Python",
                "NumPy",
                "Pandas",
                "GraphSAGE-v2",
                "Streamlit",
                "Plotly",
                "GitHub",
                "Streamlit Community Cloud"
            ],
            "Purpose": [
                "Programming language",
                "Numerical computation",
                "Dataset handling",
                "Graph-based learning model",
                "Web application framework",
                "Interactive charts",
                "Source code management",
                "Cloud deployment"
            ]
        })

        st.dataframe(
            technology_df,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        st.subheader("Project Scope")

        st.markdown("""
        This website demonstrates an experimental
        multi-sensor pipeline classification model.

        The current deployment uses prepared experimental
        samples and an exported trained model.

        Live sensor acquisition, physical sensor installation,
        real-time monitoring and edge deployment are outside
        the scope of this demonstration.
        """)

    # =================================================
    # 9. COMMON FOOTER
    # =================================================

    st.divider()

    with st.expander("ℹ️ Project Scope and Limitations"):

        st.markdown("""
        - This demonstration uses prepared experimental dataset samples.
        - The website does not currently process live sensor signals.
        - It does not provide real-time pipeline monitoring.
        - Physical sensor installation, live streaming and
          edge deployment are outside the scope of this demonstration.
        - Model performance on experimental samples may differ
          from performance in real underground pipeline environments.
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
        "GitHub repository and that the filename matches "
        "the filename used in load_model()."
    )
