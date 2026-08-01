# Subject-Conditioned Conformers on Riemannian Manifolds for Zero-Shot Generalization in Complex Dynamics

This repository contains the complete dataset and computational codebase to reproduce the findings of our study on resolving the out-of-distribution (OOD) generalization gap in highly non-stationary biological time series. By projecting complex dynamics (benchmark: continuous EEG) into a 5-dimensional (5D) phase space and applying Feature-wise Linear Modulation (FiLM) via a Subject-Conditioned Conformer, we successfully isolate a deterministic topological phase transition occurring precisely 3.0 minutes prior to the critical event.

## Overview
Existing global deep learning models for continuous physiological forecasting suffer from a profound generalization drop when applied to unseen domains (patients) due to severe inter-subject phenotypic heterogeneity. Furthermore, unconditioned models dilute predictive learning signals by evaluating performance over arbitrarily prolonged horizons.

This repository provides the code and data to demonstrate:
* The macroscopic universal collapse of network resilience at exactly 3.0 minutes prior to the non-linear cascade across unseen target domains (**Strict causal zero-shot leave-one-domain-out evaluation**).
* The microscopic mechanistic diversity of this transition, validating three distinct dynamic archetypes through high-resolution intracranial phase space tracking.

## Repository Structure
The repository is organized into three main components: Core Data, Data Extraction Pipelines, and Reproducibility/Plotting Scripts.

### 1. Core Data (.csv)
These files contain the processed features extracted from the CHB-MIT database and subsequent rigorous causal evaluation results. All evaluations strictly prohibit future-data leakage.

* `Data_Fig1_Raw_Delta_Tau.csv` : Raw tracking data containing Minimum Sample Entropy, tau_TE, tau_SVM, and Delta_tau. Used to establish topological attractor basins.
* `Data_Fig2a_Static_2D.csv` : Strict causal ROC-AUC across unseen OOD domains using an uncorrected static 2D feature model.
* `Data_Fig2b_Baseline_Tracking.csv` : Continuous tracking data demonstrating the severe spatial fragmentation of latent baseline states across distinct domains (N=7).
* `Data_Fig2c_Unconditioned_5D.csv` : Strict causal ROC-AUC using the expanded 5D feature set without dynamic baseline conditioning.
* `Data_Fig2d_Overfitting.csv` : Trajectory of Train vs. Test ROC-AUC over 20 epochs for an unconditioned model, demonstrating catastrophic overfitting to source domain distributions.
* `Data_Fig2e_Conditioned_5D_5.0min.csv` : Prediction accuracy using the FiLM-conditioned architecture evaluated at an arbitrary 5.0-minute horizon (signal dilution).
* `CHBMIT_True_FiLM_N7_Trajectory.csv` : Time-to-event ablation results, tracking mean predictive accuracy from 5.0 to 1.0 minutes prior to onset, isolating the 3.0-minute deterministic phase transition.

### 2. Data Extraction Pipelines (.py)
These scripts constitute the main computational extraction engines described in the Methods section.

* `extract_5d_features.py` : Preprocessing pipeline. Extracts 5D phase space features (Sample Entropy, Transfer Entropy via JIT-compiled estimators, and spatial classification time on Riemannian manifolds) from raw time series.
* `extract_ieeg_archetypes.py` : Pipeline to connect to the UPenn iEEG.org API, extracting the phase space collapse trajectories to validate microscopic physical mechanics (Gradual Collapse, Reflex Trigger, Late Avalanche).
* `run_true_film_n7_trajectory.py` : PyTorch implementation of the **Subject-Conditioned Conformer**. Integrates FiLM layers to dynamically scale and shift the network based on domain-specific 5D baseline vectors, executing the strict causal leave-one-domain-out (LOSO) zero-shot training and evaluation loop.

### 3. Reproducibility & Plotting Scripts (.py)
These scripts parse the finalized CSV data and generate publication-ready figures (PDF format).

* `plot_fig1_baseline_dynamics.py` : Performs clustering and statistical tests (Welch's t-test with FDR correction) to generate topological classification and attractor plots (Fig. 1).
* `plot_fig2_causal_analysis.py` : Generates the statistical plots illustrating the domain fragmentation, catastrophic overfitting, and ablation comparisons (Fig. 2a-e).
* `generate_fig3_plot_final.py` : Generates the trajectory plot identifying the macroscopic 3.0-minute deterministic phase transition (Fig. 3).
* `generate_fig4_ieeg_archetypes.py` : Generates the combined multi-trajectory plot for the three distinct microscopic dynamic archetypes (Fig. 4).

## Requirements & Environment Setup
To ensure exact reproducibility, please execute the code within a dedicated Python environment. All dependencies are listed in the provided requirements.txt.

    pip install -r requirements.txt

*(Note: The primary dependencies include pandas, numpy, scipy, scikit-learn, matplotlib, seaborn, torch, mne, numba, statsmodels, and ieeg.)*

## How to Reproduce the Main Findings

**Step 1: Verify the Data**
Ensure all `.csv` files listed above are located in the root repository directory.

**Step 2: Reproduce the Statistical Figures (Instant)**
Execute the plotting scripts directly. They will parse the pre-computed Strict Causal CSV data and output high-resolution `.pdf` files.

    python plot_fig1_baseline_dynamics.py
    python plot_fig2_causal_analysis.py
    python generate_fig3_plot_final.py
    python generate_fig4_ieeg_archetypes.py

**Step 3: (Optional) Re-run the Deep Learning Evaluation**
If you wish to re-train the Subject-Conditioned Conformer and verify the zero-shot causal predictions from scratch:

    python run_true_film_n7_trajectory.py

**Step 4: (Optional) Re-extract UPenn iEEG Data**
Ensure you have configured your iEEG.org credentials as environment variables (`IEEG_USER`, `IEEG_PWD`), then run:

    python extract_ieeg_archetypes.py

## Code Availability Policy
This repository has been established to fully comply with the Nature Portfolio policies on code and data availability, ensuring maximum transparency, reproducibility, and utility for the broader machine learning and computational neuroscience community.
