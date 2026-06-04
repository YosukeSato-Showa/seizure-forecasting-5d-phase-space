import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# --- Style & Configuration for Publication ---
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

def plot_combined_ieeg_archetypes():
    # ---------------------------------------------------------
    # 1. Generate Main Plot (Without Legend)
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6.5), dpi=300)
    
    thresholds = [5.0, 4.0, 3.0, 2.0, 1.0]

    # Extracted ground-truth AUC values
    auc_gradual   = [0.9758, 0.9886, 0.9869, 0.9833, 1.0000] # I002_P002
    auc_reflex    = [0.5824, 0.6083, 0.5515, 0.6061, 0.8121] # I002_P003
    auc_avalanche = [0.6988, 0.6636, 0.6657, 0.7470, 0.7636] # I002_P009

    # Plot trajectories and retain line objects for the legend
    line1, = ax.plot(thresholds, auc_gradual, color='#ff7f0e', marker='o', markersize=9, linewidth=3.5, 
            label='Archetype I: Gradual Collapse (P002)\nClassic secondary generalization')
            
    line2, = ax.plot(thresholds, auc_reflex, color='#2ca02c', marker='^', markersize=10, linewidth=3.5, 
            label='Archetype II: Reflex Trigger (P003)\nSudden sensory-evoked fragmentation')
            
    line3, = ax.plot(thresholds, auc_avalanche, color='#d62728', marker='s', markersize=9, linewidth=3.5, 
            label='Archetype III: Late Avalanche (P009)\nExplosive preictal spatial wave')

    # Reference lines
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1.5, zorder=0)
    ax.axvline(3.0, color='black', linestyle=':', linewidth=2)

    # Axis formatting
    ax.set_xlim(5.5, 0.5) 
    ax.set_ylim(0.45, 1.05)
    ax.set_xlabel('Preictal Threshold (Minutes prior to clinical onset)', fontweight='bold')
    ax.set_ylabel('Phase Space Collapse Severity (ROC-AUC)', fontweight='bold')
    
    ax.set_title('Figure 4: Micro-dynamic archetypes of preictal phase transitions (iEEG)', 
                 loc='left', fontweight='bold', pad=20)
    
    # Clean up spines (legend intentionally omitted)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    output_graph = 'Nature_Fig4_iEEG_Archetypes_NoLegend.pdf'
    plt.savefig(output_graph, bbox_inches='tight')
    plt.close()
    
    # ---------------------------------------------------------
    # 2. Generate Standalone Legend PDF
    # ---------------------------------------------------------
    fig_leg = plt.figure(figsize=(8, 2), dpi=300)
    ax_leg = fig_leg.add_subplot(111)
    ax_leg.axis('off') # Hide axes completely
    
    # Use handles from the main plot to create the legend
    handles = [line1, line2, line3]
    labels = [h.get_label() for h in handles]
    
    # Generate legend (no frame, center aligned)
    ax_leg.legend(handles, labels, loc='center', frameon=False, ncol=1)
    
    output_legend = 'Nature_Fig4_iEEG_Archetypes_LegendOnly.pdf'
    fig_leg.savefig(output_legend, bbox_inches='tight') # Save with minimal bounding box
    plt.close()

    print("======================================================================")
    print(f" [SUCCESS] PDF Saved (Graph Only) : {output_graph}")
    print(f" [SUCCESS] PDF Saved (Legend Only): {output_legend}")
    print("======================================================================")

if __name__ == '__main__':
    plot_combined_ieeg_archetypes()
