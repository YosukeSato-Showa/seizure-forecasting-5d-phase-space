import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
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
    'axes.labelsize': 12, 
    'axes.titlesize': 14,
    'xtick.labelsize': 11, 
    'ytick.labelsize': 11, 
    'legend.fontsize': 11, 
    'legend.frameon': False
})

def plot_box_strip(data_file: str, title: str, output_name: str, color: str):
    """
    Plots a combined boxplot and stripplot for ROC-AUC performance.
    Excludes any NaN values to maintain strict statistical integrity.
    """
    if not os.path.exists(data_file): 
        print(f"[WARNING] Data file '{data_file}' not found. Skipping plot.")
        return
        
    # Drop NaN values generated from strict causal evaluation dropouts
    df = pd.read_csv(data_file).dropna(subset=['ROC_AUC'])
    
    mean_val = df['ROC_AUC'].mean()
    median_val = df['ROC_AUC'].median()
    
    fig, ax = plt.subplots(figsize=(4.5, 6), dpi=300)
    
    # Base Boxplot
    sns.boxplot(y='ROC_AUC', data=df, color=color, width=0.4, 
                boxprops={'alpha': 0.5, 'edgecolor': 'k', 'linewidth': 1.5},
                medianprops={'color': 'k', 'linewidth': 2}, ax=ax)
                
    # Overlay Stripplot for individual patient variance
    sns.stripplot(y='ROC_AUC', data=df, color=color, size=8, alpha=0.8, 
                  edgecolor='k', linewidth=1, jitter=True, ax=ax)
    
    # Theoretical chance level
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1.5, zorder=0)
    
    # Fixed Y-axis (0.40 to 0.80) to accurately reflect the strict causal performance limits
    ax.set_ylim(0.40, 0.80)
    ax.set_ylabel('Strict Causal Prediction Accuracy (ROC-AUC)', fontweight='bold')
    ax.set_title(title, loc='left', fontweight='bold', pad=15)
    
    # Add statistical summary box
    stats_text = f"Mean (Valid N={len(df)}): {mean_val:.3f}\nMedian: {median_val:.3f}"
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9), 
            fontsize=11)
    
    sns.despine(top=True, right=True)
    plt.tight_layout()
    plt.savefig(output_name, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Saved: {output_name}")

def plot_fig2b_kde(data_file: str, output_name: str):
    """
    Plots the 2D Kernel Density Estimation (KDE) to visualize baseline 
    state space fragmentation across the entire N=7 cohort.
    """
    if not os.path.exists(data_file):
        print(f"[WARNING] Data file '{data_file}' not found. Skipping plot.")
        return
        
    df = pd.read_csv(data_file)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    
    patients = sorted(df['Patient_ID'].unique())
    palette = sns.color_palette("tab10", n_colors=len(patients))
    
    sns.scatterplot(data=df, x='tau_SVM_ms', y='tau_TE_ms', hue='Patient_ID', 
                    palette=palette, alpha=0.6, edgecolor='k', s=40, ax=ax)
    sns.kdeplot(data=df, x='tau_SVM_ms', y='tau_TE_ms', hue='Patient_ID', 
                palette=palette, fill=False, linewidths=1.5, alpha=0.8, ax=ax)
    
    ax.set_xlabel(r'Spatial Classification Time ($\tau_{SVM}$) [ms]', fontweight='bold')
    ax.set_ylabel(r'Information Transfer Delay ($\tau_{TE}$) [ms]', fontweight='bold')
    ax.set_title('b  Baseline state space fragmentation (N=7)', loc='left', fontweight='bold', pad=15)
    
    sns.despine(top=True, right=True)
    plt.legend(title='Patient ID', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_name, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Saved: {output_name}")

def plot_learning_curve(data_file: str, output_name: str):
    """
    Plots the training vs. testing AUC trajectories over epochs to demonstrate
    catastrophic overfitting in unconditioned architectures.
    """
    if not os.path.exists(data_file): 
        print(f"[WARNING] Data file '{data_file}' not found. Skipping plot.")
        return
        
    df = pd.read_csv(data_file).dropna(subset=['Test AUC (Unseen Patient)'])
    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=300)
    
    ax.plot(df['Epoch'], df['Train AUC (Source Patients)'], label='Train (Source)', 
            color='#1F77B4', linewidth=2.5, marker='o', markersize=5)
    ax.plot(df['Epoch'], df['Test AUC (Unseen Patient)'], label='Test (Target)', 
            color='#D62728', linewidth=2.5, marker='s', markersize=5)
            
    # Shade the generalization gap
    ax.fill_between(df['Epoch'], df['Train AUC (Source Patients)'], df['Test AUC (Unseen Patient)'], 
                    color='gray', alpha=0.15, label='Generalization Gap')
    
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1.5, zorder=0)
    ax.set_ylim(0.35, 1.05)
    ax.set_xticks(range(0, 21, 5))
    ax.set_xlabel('Training Epochs', fontweight='bold')
    ax.set_ylabel('ROC-AUC', fontweight='bold')
    ax.set_title('d  Overfitting to phenotypic idiosyncrasies', loc='left', fontweight='bold', pad=15)
    
    ax.legend(loc='center right')
    sns.despine(top=True, right=True)
    
    plt.tight_layout()
    plt.savefig(output_name, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Saved: {output_name}")

if __name__ == "__main__":
    print("======================================================================")
    print(" Executing Figure 2 Generation Protocol (GitHub Release Version)      ")
    print("======================================================================")
    
    # Generate all panels for Figure 2
    plot_box_strip('Data_Fig2a_Static_2D.csv', 'a  Static 2D Baseline (5.0 min)', 'Nature_Fig2a_Static_2D.pdf', '#E24A33')
    plot_fig2b_kde('Data_Fig2b_Baseline_Tracking.csv', 'Nature_Fig2b_Fragmentation.pdf')
    plot_box_strip('Data_Fig2c_Unconditioned_5D.csv', 'c  Unconditioned 5D (5.0 min)', 'Nature_Fig2c_Unconditioned_5D.pdf', '#348ABD')
    plot_learning_curve('Data_Fig2d_Overfitting.csv', 'Nature_Fig2d_Overfitting.pdf')
    plot_box_strip('Data_Fig2e_Conditioned_5D_5.0min.csv', 'e  Conditioned 5D (5.0 min)', 'Nature_Fig2e_Conditioned_5.0min.pdf', '#988ED5')
    
    print("======================================================================")
    print(" [COMPLETED] All Figure 2 PDF artifacts generated successfully.       ")
    print("======================================================================")
