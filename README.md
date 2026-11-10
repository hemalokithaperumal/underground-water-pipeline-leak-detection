# Underground Water Pipeline Leak Detection Using Multi-Sensor Signals

An experimental multi-sensor classification project that investigates how vibration, acoustic, and dynamic-pressure signals can be used to distinguish underground water-pipeline conditions. This repository contains the **Streamlit demonstration application** and its exported GraphSAGE-v2 inference model.

> **Scope note:** The web application uses prepared demonstration samples and an exported model. It is not connected to live pipeline sensors and does not perform real-time field monitoring.

## Project at a glance

- **Model in the demo:** GraphSAGE-v2
- **Sensor channels:** 6 — A1, A2, H1, H2, P1, P2
- **Features:** 16 per sensor node (96 values across six nodes)
- **Output classes:** 5 experimental pipeline-condition labels
- **Demo samples:** 60 prepared samples
- **Interface:** Streamlit with interactive Plotly visualisations

## Pipeline-condition labels

| Label | Meaning |
|---|---|
| CC | Circumferential Crack |
| GL | Gasket Leak |
| LC | Longitudinal Crack |
| NL | No Leak |
| OL | Orifice Leak |

Use the class definitions from the cited dataset documentation when interpreting experimental labels.

## Sensor channels

| Channels | Sensor type | Signal information |
|---|---|---|
| A1, A2 | Accelerometers | Pipeline vibration |
| H1, H2 | Hydrophones | Acoustic response |
| P1, P2 | Dynamic-pressure sensors | Pressure variation |

## How the demonstration works

1. Load prepared multi-sensor feature samples.
2. Standardise the sensor features using the scaler exported with the model.
3. Encode the 16 features at each of the six sensor nodes.
4. Apply GraphSAGE-v2 message-passing layers over the learned sensor graph.
5. Produce class probabilities for CC, GL, LC, NL, and OL.
6. Display the predicted class, confidence, probability distribution, and demo evaluation.

The application also lets the user select prepared signal-noise conditions. These are stored demonstration features, not signals streamed from physical sensors.

## Demonstration results

The application evaluates its **60 prepared demonstration samples** separately from the full research experiments.

- Correct predictions: **58 / 60**
- Demo accuracy: **96.67%**
- Misclassified examples: two LC samples (one predicted as OL and one as NL)

This small demonstration subset is intended to illustrate the interface and inference workflow. **Do not interpret 96.67% as the main research benchmark result or as evidence of real-world field accuracy.**

## Research evaluation context

The broader research experiments compare multiple conventional, deep-learning, and graph-learning models using grouped evaluation protocols. Selected macro-F1 results are shown below for context; they are distinct from the website's 60-sample demo score.

| Model | Unseen recordings (Test 1) | Unseen flow condition (Test 2) |
|---|---:|---:|
| GraphSAGE-v2 | **0.820** | 0.859 |
| Random Forest | 0.633 | **0.893** |

- **Test 1 — unseen recordings:** windows from the same original recording are kept together when splitting, so test recordings are not used for training.
- **Test 2 — unseen flow condition:** one flow condition is held out for testing while the model is trained on the other conditions.

These values are macro-F1 scores, not accuracy. Refer to the final research paper and its complete result tables for the full model comparison and experimental details.

## Run locally

### 1. Clone the repository

```bash
git clone https://github.com/hemalokithaperumal/underground-water-pipeline-leak-detection.git
cd underground-water-pipeline-leak-detection
```

### 2. Create and activate a virtual environment (recommended)

**Windows PowerShell**

```powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the Streamlit application

```bash
streamlit run app.py
```

The exported model file `demo_model..json` must remain in the repository root because `app.py` loads it by that exact filename.

## Repository files

| File | Purpose |
|---|---|
| `app.py` | Streamlit user interface and NumPy GraphSAGE-v2 inference |
| `demo_model..json` | Exported model parameters, scaler, class labels, and prepared demo samples |
| `requirements.txt` | Python package dependencies |

## Dataset reference

The research uses the experimental water-distribution-system dataset described in:

M. Aghashahi, L. Sela, and M. K. Banks, “Benchmarking dataset for leak detection and localization in water distribution systems,” *Data in Brief*, vol. 48, Art. no. 109148, 2023. https://doi.org/10.1016/j.dib.2023.109148

Dataset: https://doi.org/10.17632/tbrnp6vrnj.1

Please consult the dataset's licence and terms before downloading, redistributing, or reusing its raw recordings. The raw dataset is not included in this repository.

## Limitations

- The deployed interface uses prepared feature samples; it does not accept arbitrary raw sensor recordings.
- No live sensor acquisition, continuous data streaming, physical pipeline installation, or edge-device deployment is implemented here.
- Laboratory dataset performance may not transfer directly to real underground pipeline environments.
- The 60-sample demo score is not a substitute for the grouped research evaluation.

## Technologies

Python · NumPy · Pandas · Streamlit · Plotly · GraphSAGE-v2 · GitHub

## Live demonstration

[Open the Streamlit demonstration](https://underground-water-pipeline-leak-detection-uparkujfgmytvjlhu68p.streamlit.app/)

---

Developed as a final-year academic project at Saveetha Engineering College, Department of Computer Science and Engineering (Internet of Things).
