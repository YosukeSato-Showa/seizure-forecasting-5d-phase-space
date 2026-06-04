import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys
import warnings

# Suppress warnings for clean console output during automated execution
warnings.filterwarnings('ignore')

# --- 1. Style & Configuration for Publication ---
plt.style.use('seaborn-v0_8-white')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype': 42,
    'axes.linewidth': 1.5,
    'axes.labelsize': 13,
    'axes.titlesize': 15,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'legend.frameon': False
})

def plot_time_ablation(data_file: str, output_name: str) -> None:
    """
    Reads the trajectory data, computes statistical summaries (Mean, SEM), 
    and generates a clean trajectory plot isolating the phase transition.
    
    Args:
        data_file (str): Path to the input CSV containing trajectory predictions.
        output_name (str): Path/filename for the output PDF.
    """
    if not os.path.exists(data_file):
        print(f"[ERROR] Data file '{data_file}' not found.")
        sys.exit(1)
        
    df = pd.read_csv(data_file)
    
    # Aggregate statistics: Mean and Standard Error of the Mean (SEM)
    agg_df = df.groupby('Test_Min').agg(
        Mean_AUC=('AUC', 'mean'),
        std_AUC=('AUC', 'std'),
        Patients=('Patient', 'nunique')
    ).reset_index()
    
    # Calculate SEM
    agg_df['sem_AUC'] = agg_df['std_AUC'] / np.sqrt(agg_df['Patients'])
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    color_auc = '#D62728'  # Deep Red
    
    # Plot AUC Trajectory line
    ax.plot(agg_df['Test_Min'], agg_df['Mean_AUC'], color=color_auc, marker='o', 
            markersize=9, linewidth=3, label='Strict Causal Mean AUC (HQ N=7)')
    
    # Add shaded region for ± 1 SEM
    ax.fill_between(agg_df['Test_Min'], 
                    agg_df['Mean_AUC'] - agg_df['sem_AUC'], 
                    agg_df['Mean_AUC'] + agg_df['sem_AUC'], 
                    color=color_auc, alpha=0.15, label='± 1 SEM')
    
    # Theoretical chance level baseline
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1.5, zorder=0)
    
    # Critical Phase Transition Marker (3.0 minutes prior to onset)
    ax.axvline(3.0, color='k', linestyle=':', linewidth=2)
    ax.text(3.2, 0.52, 'Critical Phase Transition\n(3.0 min)', color='k', 
             fontsize=12, ha='left', va='bottom', fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))
    
    # Axis formatting
    ax.set_xlim(5.5, 0.5)  # Reversed X-axis (Approaching clinical onset)
    ax.set_ylim(0.40, 0.85)
    ax.set_xlabel('Preictal Threshold (Minutes prior to seizure onset)', fontweight='bold')
    ax.set_ylabel('Strict Causal Prediction Accuracy (ROC-AUC)', fontweight='bold')
    
    ax.set_title('Figure 3: Time-to-seizure ablation isolates a 3-minute deterministic phase transition', 
                  loc='left', fontweight='bold', pad=15)
    
    # Clean up spines for a minimalist, professional look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left', frameon=False)

    plt.tight_layout()
    plt.savefig(output_name, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Artifact saved: {output_name}")

if __name__ == "__main__":
    print("======================================================================")
    print(" Executing Figure 3 Generation Protocol (GitHub Release Version)      ")
    print("======================================================================")
    
    INPUT_CSV = 'CHBMIT_True_FiLM_N7_Trajectory.csv'
    OUTPUT_PDF = 'Nature_Fig3_TrueFiLM_N7_Trajectory_Final.pdf'
    
    plot_time_ablation(INPUT_CSV, OUTPUT_PDF)
