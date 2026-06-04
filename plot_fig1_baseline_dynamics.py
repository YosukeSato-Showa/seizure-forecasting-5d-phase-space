import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests
import os
import sys

# --- 1. Style & Configuration for Nature Communications ---
plt.style.use('seaborn-v0_8-white')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype': 42,
    'axes.linewidth': 1.2,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'legend.frameon': False
})

palette = {
    'Normal': '#2CA02C',
    'Inactive/Non Focal': '#1F77B4',
    'Active Focal': '#D62728'
}
order = ['Normal', 'Inactive/Non Focal', 'Active Focal']

# Update data file to the finalized CSV name for submission
data_file = "Data_Fig1_Raw_Delta_Tau.csv"
if not os.path.exists(data_file):
    print(f"Error: Data file '{data_file}' not found.")
    sys.exit(1)

# --- 2. Data Loading & 3-Group Classification ---
df_raw = pd.read_csv(data_file)
numeric_cols = ['Min_SampEn', 'tau_TE_ms', 'tau_SVM_ms', 'Delta_Tau_ms', 'Max_SVM_Acc']
df_patients = df_raw.groupby(['Patient_ID', 'Label'])[numeric_cols].mean().reset_index()

# Clean outliers for consistency
df_clean = df_patients[(df_patients['Delta_Tau_ms'] >= -50) & (df_patients['Delta_Tau_ms'] <= 50)].copy()

# Stratify
normal = df_clean[df_clean['Label'] == 'Normal'].copy()
epilepsy = df_clean[df_clean['Label'] == 'Epilepsy'].copy()

# K-Means on Min_SampEn
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
epilepsy['Cluster'] = kmeans.fit_predict(epilepsy[['Min_SampEn']])
cluster_means = epilepsy.groupby('Cluster')['Min_SampEn'].mean()
active_cluster_id = cluster_means.idxmin()

active_focal = epilepsy[epilepsy['Cluster'] == active_cluster_id].copy()
inactive_non_focal = epilepsy[epilepsy['Cluster'] != active_cluster_id].copy()

active_focal['Group'] = 'Active Focal'
inactive_non_focal['Group'] = 'Inactive/Non Focal'
normal['Group'] = 'Normal'

df_3groups = pd.concat([normal, active_focal, inactive_non_focal])

# --- 3. Statistical Analysis for tau_SVM ---
# Descriptive Stats
stats_df = df_3groups.groupby('Group')['tau_SVM_ms'].agg(['count', 'mean', 'std', 'median']).reindex(order)

# Welch's t-test
g_n = normal['tau_SVM_ms']
g_i = inactive_non_focal['tau_SVM_ms']
g_a = active_focal['tau_SVM_ms']

p_n_vs_a = stats.ttest_ind(g_n, g_a, equal_var=False)[1]
p_n_vs_i = stats.ttest_ind(g_n, g_i, equal_var=False)[1]
p_i_vs_a = stats.ttest_ind(g_i, g_a, equal_var=False)[1]

# FDR Correction
p_vals = [p_n_vs_a, p_n_vs_i, p_i_vs_a]
reject, pvals_corrected, _, _ = multipletests(p_vals, alpha=0.05, method='fdr_bh')

# Append results to stats DataFrame
comparison_results = pd.DataFrame({
    'Comparison': ['Normal vs Active Focal', 'Normal vs Inactive', 'Inactive vs Active Focal'],
    'Raw_p_value': p_vals,
    'FDR_p_value': pvals_corrected
})

# Save stats to CSV
stats_df.to_csv("Fig1b_tauSVM_Descriptive_Stats.csv")
comparison_results.to_csv("Fig1b_tauSVM_Statistical_Tests.csv", index=False)

print("Statistical results saved to CSV files.")

# --- 4. Plotting Figure 1a (tau_SVM Boxplot) ---
fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

sns.boxplot(data=df_3groups, x='Group', y='tau_SVM_ms', order=order, palette=palette, 
            boxprops={'alpha': 0.7, 'edgecolor': 'k', 'linewidth': 1.5}, 
            medianprops={'color': 'k', 'linewidth': 2}, ax=ax)
sns.stripplot(data=df_3groups, x='Group', y='tau_SVM_ms', order=order, palette=palette, 
              dodge=True, size=5, alpha=0.6, edgecolor='k', linewidth=0.5, ax=ax)

# Formatting
ax.set_title(r'a  Spatial classification time ($\tau_{SVM}$) across cohorts', loc='left', fontweight='bold', pad=15)
ax.set_ylabel(r'$\tau_{SVM}$ (ms)')
ax.set_xlabel('')

# Annotate significance (if any) or 'ns' (not significant)
def sig_marker(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

y_max = df_3groups['tau_SVM_ms'].max()
y_range = y_max - df_3groups['tau_SVM_ms'].min()
h = y_range * 0.05

# Annotate Normal vs Active Focal
ax.plot([0, 0, 2, 2], [y_max + h, y_max + 2*h, y_max + 2*h, y_max + h], lw=1.2, c='k')
ax.text(1, y_max + 2.2*h, sig_marker(pvals_corrected[0]), ha='center', va='bottom', color='k', fontsize=12)

# Clean up axes
sns.despine(top=True, right=True)
ax.set_ylim(bottom=0, top=y_max + 4*h)

plt.tight_layout()
fig_filename = 'Nature_Fig_1a_tauSVM_Boxplot.pdf'
plt.savefig(fig_filename, bbox_inches='tight')
plt.close(fig)

print(f"Figure saved successfully: {fig_filename}")