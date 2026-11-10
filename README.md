# Deep Learning-Based Detection and Classification of Underground Water Pipeline Leaks Using Multi-Sensor Signals

An experimental research project that studies how vibration, acoustic, and dynamic-pressure signals can help distinguish pipeline conditions in a laboratory water-distribution testbed. This repository contains the Streamlit demonstration application, its exported GraphSAGE-v2 inference model, and the research notebook.

> **Scope:** This is a laboratory-dataset research prototype. The website runs inference on prepared feature samples. It is not connected to live pipeline sensors and does not provide real-time field monitoring.

## Project overview

| Item | Details |
|---|---|
| Research task | Five-class pipeline-condition classification |
| Dataset recordings | 40 |
| Windowed samples | 2,360 |
| Window size / hop | 1.0 second / 0.5 second |
| Sensor channels | 6: A1, A2, H1, H2, P1, P2 |
| Main model input | 16 features per sensor node (96 values per window) |
| Main graph model | GraphSAGE-v2 |
| Website | Streamlit + Plotly |

## Dataset and labels

The experiments use the laboratory testbed dataset described by Aghashahi, Sela, and Banks (2023). The notebook selects 40 recordings across two network configurations (BR and LO), five condition labels, and four operating conditions. It uses 30 seconds from each recording.

| Label | Dataset condition name |
|---|---|
| CC | Circumferential Crack |
| GL | Gasket Leak |
| LC | Longitudinal Crack |
| NL | No Leak |
| OL | Orifice Leak |

The six sensor channels are:

| Channels | Sensor type | Measurement |
|---|---|---|
| A1, A2 | Accelerometers | Pipeline vibration |
| H1, H2 | Hydrophones | Acoustic response |
| P1, P2 | Dynamic-pressure sensors | Pressure variation |

The raw dataset is not included in this repository. Please consult the dataset's licence and terms before downloading, redistributing, or reusing the recordings.

## Preprocessing and features

- The notebook loads accelerometer and pressure CSV files and hydrophone RAW recordings.
- Hydrophone RAW data are read as little-endian 32-bit integer samples and scaled by (2^{31}), using the 8 kHz sampling rate and the first 240,000 samples (30 seconds).
- Signals are divided into 1-second windows with a 0.5-second hop. This produces 59 windows per recording and 2,360 windows overall.
- Each sensor node has 16 features: total log spectral power, 12 log spectral-band power fractions, skewness, kurtosis, and crest factor.
- The six sensor nodes form a (6 \times 16) feature tensor (96 feature values per window).

The feature-engineering cell also calculates pairwise features for A1–A2, H1–H2, and P1–P2. These are used in a preliminary Random Forest comparison. The later neural-network and graph-model benchmark uses the six-node, 16-feature-per-node tensor; it does not concatenate those pairwise features into the final GraphSAGE-v2 input.

## Evaluation protocols

**Test 1 — unseen recordings:** StratifiedGroupKFold splits by recording ID. Windows from one recording stay in the same fold, reducing the risk of overlapping windows from one recording appearing in both training and test sets.

**Test 2 — unseen flow condition:** leave-one-operating-condition-out evaluation. Each of the four operating conditions is held out in turn, and the other three are used for training.

Scores below are mean **macro-F1** across folds. Macro-F1 gives equal weight to each class; it is not accuracy.

## Model comparison

| Model | Test 1: unseen recordings | Test 2: unseen flow condition |
|---|---:|---:|
| **GraphSAGE-v2** | **0.820** | 0.859 |
| GCN-v2 (learned edges) | 0.729 | 0.856 |
| GCN-v2 (uniform edges) | 0.724 | 0.825 |
| No-graph ablation | 0.708 | 0.882 |
| CNN-LSTM | 0.701 | 0.826 |
| Fusion-CNN | 0.683 | 0.870 |
| **Random Forest** | 0.633 | **0.893** |
| BiLSTM | 0.612 | 0.749 |
| 1D-CNN | 0.610 | 0.778 |
| GraphSAGE (v1) | 0.597 | 0.714 |
| GAT-v2 | 0.597 | 0.794 |
| GCN (v1) | 0.468 | 0.636 |
| GAT (v1) | 0.431 | 0.550 |

**Interpretation:** GraphSAGE-v2 achieves the highest macro-F1 in Test 1. Random Forest achieves the highest macro-F1 in Test 2. Therefore, GraphSAGE-v2 is a strong result for the unseen-recording protocol, but it is not the best model under every evaluation protocol.

## Noise robustness experiments

The notebook adds controlled Gaussian noise to the raw test signals and reports macro-F1 at several signal-to-noise ratios (SNRs). The following results are for GraphSAGE-v2.

| Evaluation | Training setup | Clean | 20 dB | 10 dB | 5 dB | 0 dB | -5 dB |
|---|---|---:|---:|---:|---:|---:|---:|
| Test 1 | Clean only | 0.820 | 0.439 | 0.228 | 0.183 | 0.152 | 0.134 |
| Test 1 | Clean + noisy copies | 0.692 | 0.673 | 0.647 | 0.634 | 0.548 | 0.379 |
| Test 2 | Clean only | 0.859 | 0.564 | 0.258 | 0.193 | 0.124 | 0.087 |
| Test 2 | Clean + noisy copies | 0.878 | 0.887 | 0.843 | 0.846 | 0.794 | 0.476 |

The noise-augmented training experiment uses 20 dB and 5 dB noisy copies in the training set. It improves performance under many noisy test conditions, while clean-signal performance changes. The -5 dB condition remains challenging.

## Conditional-GAN experiment

The notebook also tests conditional-GAN-generated feature samples as an additional augmentation method.

For GraphSAGE-v2 at the clean test level:

| Evaluation | Clean only | Clean + GAN | Noise augmentation | Noise augmentation + GAN |
|---|---:|---:|---:|---:|
| Test 1 | 0.820 | 0.650 | 0.692 | 0.596 |
| Test 2 | 0.859 | 0.777 | 0.878 | 0.768 |

In these experiments, adding GAN-generated samples did not improve GraphSAGE-v2 over noise-only augmentation. The GAN results should be presented as a negative or mixed experimental finding, not as a performance improvement.

## Streamlit demonstration

The website uses a separately exported GraphSAGE-v2 model trained on three operating conditions and evaluated on the held-out **0.47 LPS** condition.

- Training windows: 1,770
- Held-out windows: 590
- Clean held-out accuracy: 96.27%
- Clean held-out macro-F1: 0.9626

| Held-out test signal | Accuracy | Macro-F1 |
|---|---:|---:|
| Clean | 96.27% | 0.9626 |
| 20 dB | 96.10% | 0.9608 |
| 10 dB | 96.27% | 0.9624 |
| 5 dB | 95.59% | 0.9550 |
| 0 dB | 91.86% | 0.9153 |
| -5 dB | 73.22% | 0.7360 |

The Streamlit app also displays 60 selected prepared examples from the held-out condition. In the clean 60-sample subset, 58 predictions are correct (96.67%) and two LC samples are misclassified (one as OL and one as NL). This subset result is not the same as the 590-window held-out evaluation or the cross-validation benchmark.

**Demo limitations:** the app accepts prepared feature samples only. It does not ingest arbitrary raw recordings, connect to physical sensors, stream live data, or perform field monitoring. Experimental testbed results do not establish real-world field accuracy.

## Run the website locally

### 1. Clone the repository

```bash
git clone https://github.com/hemalokithaperumal/underground-water-pipeline-leak-detection.git
cd underground-water-pipeline-leak-detection
```

### 2. Create a virtual environment (recommended)

**Windows PowerShell**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the website dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

The exported model file is named `demo_model..json` (with two dots) because the current app loads that exact filename. Keep it in the repository root unless you also update the path in `app.py`.

## Research notebook

Open [`notebooks/research_experiments.ipynb`](notebooks/research_experiments.ipynb). It is a Google Colab workflow that reads the raw dataset from Google Drive. Before running it, download the dataset, update the `ROOT` paths to your Drive folder, and execute the cells in order. Intermediate files are written to `processed_real` in Drive and are not committed to this repository.

The notebook documents dataset inspection, feature extraction, model benchmarking, noise robustness, GAN augmentation, and export of the separate website model.

## Repository structure

```text
.
├── app.py
├── demo_model..json
├── requirements.txt
├── README.md
├── .gitignore
└── notebooks/
    └── research_experiments.ipynb
```

| File | Purpose |
|---|---|
| `app.py` | Streamlit interface and NumPy implementation of GraphSAGE-v2 inference |
| `demo_model..json` | Exported model weights, scaler, class labels, held-out metrics, and prepared demo samples |
| `requirements.txt` | Dependencies required to run the website |
| `notebooks/research_experiments.ipynb` | Google Colab research and model-export workflow |
| `.gitignore` | Excludes local environments, caches, and generated data artifacts |

## Dataset citation

M. Aghashahi, L. Sela, and M. K. Banks, “Benchmarking dataset for leak detection and localization in water distribution systems,” *Data in Brief*, vol. 48, Art. no. 109148, 2023. https://doi.org/10.1016/j.dib.2023.109148

Dataset: https://doi.org/10.17632/tbrnp6vrnj.1

Please follow the dataset provider's licence and terms for access and reuse.

## Technologies

Python · NumPy · Pandas · SciPy · scikit-learn · PyTorch · Streamlit · Plotly · Google Colab · GitHub

## Live demonstration

[Open the Streamlit demonstration](https://underground-water-pipeline-leak-detection-uparkujfgmytvjlhu68p.streamlit.app/)

---

Academic final-year project, Saveetha Engineering College, Department of Computer Science and Engineering (Internet of Things).
