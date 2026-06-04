# Neurophysiologically Grounded Seizure Forecasting: A 5D Phase Space Approach

This repository contains the complete dataset and computational codebase to reproduce the findings of our study on personalized seizure prediction. By projecting electroencephalogram (EEG) signals into a **5-dimensional (5D) phase space** and applying **Feature-wise Linear Modulation (FiLM)**, we successfully isolate a deterministic preictal phase transition occurring precisely 3.0 minutes prior to seizure onset.


## Overview
Existing computational models for seizure forecasting suffer from a profound performance drop when applied to unseen patients due to inter-subject phenotypic heterogeneity. Furthermore, traditional models evaluate performance using arbitrarily prolonged preictal windows (e.g., 15–60 minutes), which dilutes the predictive learning signal due to intact surround inhibition.

This repository provides the code and data to demonstrate:
1. The macroscopic universal collapse of surround inhibition at exactly 3.0 minutes prior to clinical onset across a cohort of unseen patients (Strict Causal LOSO evaluation on Scalp EEG).
2. The microscopic mechanistic diversity of this collapse, revealing three distinct epileptogenesis archetypes through intracranial EEG (iEEG) phase space tracking.

---

## Repository Structure

The repository is organized into three main components: **Core Data**, **Data Extraction Pipelines**, and **Reproducibility/Plotting Scripts**.

### 1. Core Data (`.csv`)
These files contain the processed features extracted from the CHB-MIT Scalp EEG database and subsequent rigorous causal evaluation results. All evaluations strictly prohibit future-data leakage.
* `Data_Fig1_Raw_Delta_Tau.csv` : Raw tracking data containing Minimum Sample Entropy, $\tau_{TE}$, $\tau_{SVM}$, and $\Delta\tau$ across Normal, Inactive, and Active Focal epochs. Used to establish topological attractors.
* `Data_Fig2a_Static_2D.csv` : Strict Causal ROC-AUC across unseen patients using an uncorrected static 2D feature model (Evaluated at 5.0 minutes prior to onset).
* `Data_Fig2b_Baseline_Tracking.csv` : Continuous tracking data demonstrating the severe spatial fragmentation of baseline states across individuals (N=7).
* `Data_Fig2c_Unconditioned_5D.csv` : Strict Causal ROC-AUC using the expanded 5D feature set *without* dynamic baseline conditioning (5.0 minutes prior).
* `Data_Fig2d_Overfitting.csv` : Trajectory of Train vs. Test ROC-AUC over 20 epochs for an unconditioned model, demonstrating catastrophic overfitting to phenotypic idiosyncrasies.
* `Data_Fig2e_Conditioned_5D_5.0min.csv` : Prediction accuracy using the FiLM-conditioned architecture evaluated at a 5.0-minute horizon.
* `CHBMIT_True_FiLM_N7_Trajectory.csv` : Time-to-seizure ablation results, tracking mean predictive accuracy from 5.0 to 1.0 minutes prior to clinical onset, isolating the 3.0-minute phase transition.

### 2. Data Extraction Pipelines (`.py`)
These scripts constitute the main computational extraction engines described in the Methods section.
* `extract_5d_features.py` : MNE-Python based preprocessing pipeline. Extracts 5D phase space features (Sample Entropy, Transfer Entropy via JIT-compiled estimators, and spatial classification time) from raw CHB-MIT `.edf` files.
* `extract_ieeg_archetypes.py` : Unified pipeline to connect to the UPenn iEEG.org portal, extracting the phase space collapse trajectories for three specific mechanistic archetypes (Gradual Collapse, Reflex Trigger, Late Avalanche).
* `run_true_film_n7_trajectory.py` : PyTorch implementation of the **Subject-Conditioned Conformer**. Integrates FiLM layers to dynamically scale and shift the network based on patient-specific 5D baseline vectors, executing the strict causal leave-one-subject-out (LOSO) training and evaluation loop.

### 3. Reproducibility & Plotting Scripts (`.py`)
These scripts parse the finalized CSV data and generate publication-ready figures (PDF format) adhering to *Nature Communications* style guidelines.
* `plot_fig1_baseline_dynamics.py` : Performs clustering and statistical tests (Welch's t-test with FDR correction) to generate spatial classification time boxplots (Fig. 1).
* `plot_fig2_causal_analysis.py` : Generates the statistical plots illustrating the generalization gap, spatial fragmentation, and 5.0-minute ablation comparisons (Fig. 2a-e).
* `generate_fig3_plot_final.py` : Generates the clean, single-axis trajectory plot identifying the macroscopic 3.0-minute deterministic phase transition (Fig. 3).
* `generate_fig4_ieeg_archetypes.py` : Generates the combined multi-trajectory plot for the three iEEG epileptogenesis archetypes (Fig. 4).

---

## Requirements & Environment Setup

To ensure exact reproducibility, please execute the code within a dedicated Python environment. All dependencies are listed in the provided `requirements.txt`.

```bash
# It is recommended to use a virtual environment (e.g., conda or venv)
pip install -r requirements.txt
```

*(Note: The primary dependencies include pandas, numpy, scipy, scikit-learn, matplotlib, seaborn, torch, mne, numba, statsmodels, and ieeg.)*

---

## How to Reproduce the Main Findings

**Step 1: Verify the Data**
Ensure all `.csv` files listed above are located in the root repository directory.

**Step 2: Reproduce the Statistical Figures (Instant)**
Execute the plotting scripts directly. They will parse the pre-computed Strict Causal CSV data and output high-resolution `.pdf` files.

```bash
python plot_fig1_baseline_dynamics.py
python plot_fig2_causal_analysis.py
python generate_fig3_plot_final.py
python generate_fig4_ieeg_archetypes.py
```

**Step 3: (Optional) Re-run the Deep Learning Evaluation**
If you wish to re-train the Subject-Conditioned Conformer and verify the causal predictions from scratch:

```bash
python run_true_film_n7_trajectory.py
```

**Step 4: (Optional) Re-extract UPenn iEEG Data**
Ensure you have configured your iEEG.org credentials as environment variables (`IEEG_USER`, `IEEG_PWD`), then run:

```bash
python extract_ieeg_archetypes.py
```

---

## Code Availability Policy

This repository has been established to fully comply with the *Nature Portfolio* policies on code and data availability, ensuring maximum transparency, reproducibility, and utility for the broader computational neuroscience community.
