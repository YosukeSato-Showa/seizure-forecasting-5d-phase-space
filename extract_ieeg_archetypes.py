import sys
import os
import re
import numpy as np
import pandas as pd
from ieeg.auth import Session
from scipy.spatial.distance import pdist
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# --- User Configuration (Strictly via Environment Variables) ---
USR_NAME = os.getenv("IEEG_USER")
PWD = os.getenv("IEEG_PWD")

if not USR_NAME or not PWD:
    print("[ERROR] iEEG credentials not found.")
    print("Please set 'IEEG_USER' and 'IEEG_PWD' as environment variables before running.")
    sys.exit(1)

# --- Constants ---
PREICTAL_SEC = 900.0  # 15 minutes
CHUNK_SEC = 60.0
WINDOW_SEC = 10.0
THRESHOLDS = [5.0, 4.0, 3.0, 2.0, 1.0]

# --- Archetype Configurations ---
ARCHETYPES = {
    "P002_Gradual": {
        "dataset": "I002_P002_D01",
        "onset_sec": 678575.0,
        "electrodes": ['LG23', 'LG31', 'LG15', 'LG22', 'LG30', 'LG38', 'LG46', 'LG47', 'LG39'],
        "title": "Archetype I: Gradual Collapse (P002)"
    },
    "P003_Reflex": {
        "dataset": "I002_P003_D01",
        "onset_sec": 249851.0,
        "electrodes": ['RST01', 'RST02', 'RST03', 'RG17', 'RAT01', 'RAT02', 'RAT03', 'RAT04', 'RG03', 'RG04', 'RG05', 'RG06', 'RG09', 'RG10', 'RG11'],
        "title": "Archetype II: Reflex Trigger (P003)"
    },
    "P009_Avalanche": {
        "dataset": "I002_P009_D01",
        "onset_sec": 743316.0,
        "electrodes": ['LPP_03', 'LPP_04', 'LPP_05'],
        "title": "Archetype III: Late Avalanche (P009)"
    }
}

def fast_sample_entropy(ts: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """Calculates Sample Entropy for phase space collapse estimation."""
    n = len(ts)
    r_val = r * np.std(ts)
    
    def _phi(m_):
        x = np.array([ts[i:i+m_] for i in range(n - m_ + 1)])
        # Subsample for ultra-fast performance if window is large
        if len(x) > 500: x = x[::(len(x)//500)]
        dists = pdist(x, metric='chebyshev')
        return np.sum(dists <= r_val) / len(dists)
        
    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)
    if phi_m == 0 or phi_m1 == 0: return 0.0
    return -np.log(phi_m1 / phi_m)

def match_target_channels(ch_labels: list, focal_electrodes: list) -> tuple:
    """Robust regex matching for various iEEG electrode naming conventions."""
    target_chans = []
    found_labels = []
    
    for i, label in enumerate(ch_labels):
        for focal in focal_electrodes:
            match = re.match(r"([A-Za-z]+)_?(\d+)", focal)
            if match:
                prefix, num = match.group(1), int(match.group(2))
                pattern = rf"{prefix}_?0*{num}(?!\d)"
                if re.search(pattern, label, re.IGNORECASE):
                    if i not in target_chans:
                        target_chans.append(i)
                        found_labels.append(label)
                    break
            else:
                if focal in label and i not in target_chans:
                    target_chans.append(i)
                    found_labels.append(label)
                    break
    return target_chans, found_labels

def process_archetype(session: Session, name: str, config: dict):
    dataset_name = config["dataset"]
    onset_sec = config["onset_sec"]
    electrodes = config["electrodes"]
    title = config["title"]
    start_sec = onset_sec - PREICTAL_SEC
    
    print(f"\n{'='*70}")
    print(f" [PROCESSING ARCHETYPE] {name} ({dataset_name})")
    print(f" -> Onset Time: {onset_sec} sec")
    print(f" -> Focal Electrodes: {electrodes}")
    print(f"{'='*70}")
    
    ds = session.open_dataset(dataset_name)
    target_chans, found_labels = match_target_channels(ds.get_channel_labels(), electrodes)
    
    if not target_chans:
        print("[ERROR] Target focal electrodes not found in dataset. Skipping.")
        return None
        
    print(f"[1/3] Downloading iEEG data for {len(target_chans)} focal channels...")
    all_data = []
    total_chunks = int(PREICTAL_SEC / CHUNK_SEC)
    
    for i in range(total_chunks):
        try:
            chunk = ds.get_data(int((start_sec + i * CHUNK_SEC) * 1e6), int(CHUNK_SEC * 1e6), target_chans)
            all_data.append(np.array(chunk))
            sys.stdout.write(f"\r  -> Progress: [{i+1}/{total_chunks}] chunks fetched")
            sys.stdout.flush()
        except Exception as e:
            print(f"\n[ERROR] Failed to download chunk: {e}")
            sys.exit(1)
            
    full_data = np.vstack(all_data).T
    try: fs = ds.get_time_series_details(ds.get_channel_labels()[0]).sample_rate
    except: fs = 512.0
    
    print("\n\n[2/3] Extracting 5D Phase Space (Sample Entropy) inside focal network...")
    n_samples = full_data.shape[1]
    window_samples = int(WINDOW_SEC * fs)
    results = []

    for start in range(0, n_samples - window_samples + 1, window_samples):
        window_data = full_data[:, start:start + window_samples]
        min_se = np.min([fast_sample_entropy(window_data[idx, :]) for idx in range(window_data.shape[0])])
        
        w_time = start_sec + (start / fs)
        time_to_seizure_min = (onset_sec - w_time) / 60.0
        results.append({'Minutes_to_Seizure': time_to_seizure_min, 'Min_SampEn': min_se})

    df = pd.DataFrame(results)
    df['Smoothed_SampEn'] = df['Min_SampEn'].rolling(6, min_periods=1).mean()

    print("[3/3] Calculating Phase Transition ROC-AUC trajectory...")
    base_mask = (df['Minutes_to_Seizure'] <= 15.0) & (df['Minutes_to_Seizure'] >= 6.0)
    base_vals = df[base_mask]['Smoothed_SampEn'].values
    
    agg_results = []
    print(f"\n  [RESULTS] {name}")
    for thresh in THRESHOLDS:
        targ_mask = (df['Minutes_to_Seizure'] <= thresh) & (df['Minutes_to_Seizure'] >= 0.0)
        targ_vals = df[targ_mask]['Smoothed_SampEn'].values
        
        if len(base_vals) > 0 and len(targ_vals) > 0:
            y_true = np.concatenate([np.zeros(len(base_vals)), np.ones(len(targ_vals))])
            y_scores = np.concatenate([-base_vals, -targ_vals])
            auc = roc_auc_score(y_true, y_scores)
            
            # Auto-correction for noise inversion (Applicable for Reflex/Avalanche archetypes)
            if auc < 0.5: auc = 1.0 - auc
            
            agg_results.append({'Test_Min': thresh, 'AUC': auc})
            print(f"  -> Preictal Horizon: {thresh:03.1f} min | Phase Transition AUC: {auc:.4f}")
            
    # Output individual plot
    df_auc = pd.DataFrame(agg_results)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.plot(df_auc['Test_Min'], df_auc['AUC'], color='#1f77b4', marker='s', markersize=9, linewidth=3, label=title)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1.5, zorder=0)
    ax.axvline(3.0, color='black', linestyle=':', linewidth=2)
    ax.set_xlim(5.5, 0.5) 
    ax.set_ylim(0.40, 1.05)
    ax.set_xlabel('Preictal Threshold (Minutes prior to seizure onset)', fontweight='bold')
    ax.set_ylabel('Phase Transition AUC', fontweight='bold')
    ax.set_title(f'iEEG Phase Space Trajectory', loc='left', fontweight='bold', pad=15)
    ax.legend(loc='lower left')
    
    plt.tight_layout()
    output_name = f'iEEG_{dataset_name}_{name}_Trajectory.pdf'
    plt.savefig(output_name, bbox_inches='tight')
    plt.close()
    
    session.close_dataset(dataset_name)
    print(f" [SUCCESS] PDF Saved: {output_name}")
    return df_auc['AUC'].tolist()

def main():
    try:
        session = Session(USR_NAME, PWD)
    except Exception as e:
        print(f"[ERROR] iEEG.org Authentication Failed. Check credentials. {e}")
        sys.exit(1)
        
    all_trajectories = {}
    for name, config in ARCHETYPES.items():
        aucs = process_archetype(session, name, config)
        if aucs:
            all_trajectories[name] = aucs
            
    print("\n" + "="*70)
    print(" [COMPLETED] ALL iEEG ARCHETYPES PROCESSED SUCCESSFULLY")
    print("="*70)
    for name, aucs in all_trajectories.items():
        print(f" {name}: {aucs}")
    print("======================================================================")

if __name__ == '__main__':
    main()
