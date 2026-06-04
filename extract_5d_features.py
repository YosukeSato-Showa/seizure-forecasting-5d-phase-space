import mne
import numpy as np
import glob
import os
import pandas as pd
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from numba import njit, prange
import warnings
import csv

warnings.filterwarnings('ignore')

# --- Parameter Settings ---
LAG_SAMPLES = np.array([1, 3, 5, 7, 10, 15, 20, 25], dtype=np.int32)
WINDOW_SEC = 10.0
STEP_SEC = 2.0
FS = 256
WINDOW_SIZE = int(WINDOW_SEC * FS)
STEP_SIZE = int(STEP_SEC * FS)

# Relative path configuration for repository portability
DATA_DIR = "./CHBMIT_Data"
OUTPUT_CSV = "Data_Fig1_Raw_Delta_Tau.csv"

TARGET_ELECTRODES = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ']

SEIZURE_DICT = {
    "chb01_03.edf": [(2996, 3036)], "chb01_04.edf": [(1467, 1494)], "chb01_15.edf": [(1732, 1772)],
    "chb01_16.edf": [(1015, 1066)], "chb01_18.edf": [(1720, 1810)], "chb01_21.edf": [(327, 420)],
    "chb01_26.edf": [(1862, 1963)], "chb02_16.edf": [(130, 212)], "chb02_19.edf": [(3369, 3378)],
    "chb03_01.edf": [(362, 414)], "chb03_02.edf": [(731, 796)], "chb03_03.edf": [(432, 501)],
    "chb03_04.edf": [(2162, 2214)], "chb03_34.edf": [(1982, 2029)], "chb03_35.edf": [(2592, 2656)],
    "chb03_36.edf": [(1725, 1778)], "chb04_05.edf": [(7804, 7853)], "chb04_08.edf": [(6446, 6557)],
    "chb04_28.edf": [(1679, 1781), (3782, 3898)], "chb05_06.edf": [(417, 532)], "chb05_13.edf": [(1086, 1196)],
    "chb05_16.edf": [(2317, 2413)], "chb05_17.edf": [(2451, 2571)], "chb05_22.edf": [(2348, 2465)],
    "chb06_01.edf": [(1724, 1738), (7461, 7476), (13525, 13540)], "chb07_12.edf": [(4920, 5006)],
    "chb07_13.edf": [(3285, 3381)], "chb07_19.edf": [(13688, 13831)], "chb08_02.edf": [(2670, 2841)],
    "chb08_05.edf": [(2856, 3046)], "chb08_11.edf": [(2988, 3122)], "chb08_13.edf": [(2417, 2577)],
    "chb08_21.edf": [(2083, 2347)], "chb10_12.edf": [(6313, 6348)], "chb15_06.edf": [(272, 397)],
    "chb16_10.edf": [(2290, 2299)], "chb20_13.edf": [(1405, 1445)], "chb22_20.edf": [(3367, 3425)]
}

# ==========================================
# Accelerated Functions Compiled with Numba
# ==========================================

@njit(fastmath=True)
def calc_sampen_fast(signal, m=2, r=0.2):
    """ Ultra-fast Sample Entropy calculation using Numba JIT """
    n = len(signal)
    B = 0
    A = 0
    for i in range(n - m):
        for j in range(i + 1, n - m):
            dist_m = 0.0
            for k in range(m):
                d = abs(signal[i+k] - signal[j+k])
                if d > dist_m: dist_m = d
            if dist_m <= r:
                B += 1
                d_m1 = abs(signal[i+m] - signal[j+m])
                if max(dist_m, d_m1) <= r:
                    A += 1
    if B > 0 and A > 0:
        return -np.log(A / float(B))
    return np.nan

@njit(fastmath=True)
def calc_te_fast(x, y, lag):
    """ Ultra-fast Transfer Entropy calculation using Numba JIT """
    n = len(x)
    y_t = y[lag:]
    y_tm_lag = y[:-lag]
    x_tm_lag = x[:-lag]
    
    # High-speed array-based processing for state counts
    counts_joint = np.zeros(8)
    counts_yx = np.zeros(4)
    counts_yy = np.zeros(4)
    counts_y1 = np.zeros(2)
    
    length = n - lag
    for i in range(length):
        st_joint = y_t[i]*4 + y_tm_lag[i]*2 + x_tm_lag[i]
        st_yx = y_tm_lag[i]*2 + x_tm_lag[i]
        st_yy = y_t[i]*2 + y_tm_lag[i]
        st_y1 = y_tm_lag[i]
        
        counts_joint[st_joint] += 1
        counts_yx[st_yx] += 1
        counts_yy[st_yy] += 1
        counts_y1[st_y1] += 1
        
    te = 0.0
    for i in range(8):
        if counts_joint[i] > 0:
            yt = (i // 4) % 2
            ytm_lag = (i // 2) % 2
            xtm_lag = i % 2
            
            p_joint = counts_joint[i] / length
            p_y_given_yx = counts_joint[i] / (counts_yx[ytm_lag * 2 + xtm_lag] + 1e-9)
            p_y_given_y = counts_yy[yt * 2 + ytm_lag] / (counts_y1[ytm_lag] + 1e-9)
            
            if p_y_given_y > 0 and p_y_given_yx > 0:
                te += p_joint * np.log2(p_y_given_yx / p_y_given_y)
    return te

def binarize_signal(signal):
    """ Binarize signal based on median threshold """
    return np.where(signal > np.median(signal), 1, 0).astype(np.int32)

# ==========================================

def process_continuous_window(start_idx, data, times, signals, target_electrodes, seizure_intervals):
    end_idx = start_idx + WINDOW_SIZE
    current_time_sec = times[start_idx]
    
    phase_label = 'Baseline'
    time_to_seizure = None
    
    is_near_seizure = False
    for (s_start, s_end) in seizure_intervals:
        if s_start <= current_time_sec <= s_end:
            phase_label = 'Ictal'
            time_to_seizure = 0
            is_near_seizure = True
            break
        elif 0 < (s_start - current_time_sec) <= 1800:
            phase_label = 'Pre-ictal'
            time_to_seizure = s_start - current_time_sec
            is_near_seizure = True
            break
        elif 0 < (current_time_sec - s_end) <= 120:
            phase_label = 'Post-ictal'
            time_to_seizure = -1
            is_near_seizure = True
            break
            
    if not is_near_seizure and all((s_start - current_time_sec) > 900 for (s_start, s_end) in seizure_intervals):
        return None
            
    # Phase 0: Driver Node (JIT Accelerated)
    sampen_scores = {}
    for ch in target_electrodes:
        sig = signals[ch][start_idx:end_idx]
        r_val = 0.2 * np.std(sig)
        se = calc_sampen_fast(sig, 2, r_val)
        if not np.isnan(se): sampen_scores[ch] = se
    
    if not sampen_scores: return None
    driver_ch = min(sampen_scores, key=sampen_scores.get)
    min_se = sampen_scores[driver_ch]
    
    # Phase 1: TE (JIT Accelerated)
    signals_bin = {k: binarize_signal(v[start_idx:end_idx]) for k, v in signals.items()}
    best_te_drive = -999
    tau_TE_sample = 1
    
    driver_bin = signals_bin[driver_ch]
    for lag in LAG_SAMPLES:
        total_outflow, total_inflow = 0.0, 0.0
        for e_tgt in target_electrodes:
            if driver_ch == e_tgt: continue
            tgt_bin = signals_bin[e_tgt]
            total_outflow += calc_te_fast(driver_bin, tgt_bin, lag)
            total_inflow += calc_te_fast(tgt_bin, driver_bin, lag)
        net_drive = total_outflow - total_inflow
        if net_drive > best_te_drive:
            best_te_drive = net_drive
            tau_TE_sample = lag
            
    tau_TE_ms = (tau_TE_sample / FS) * 1000

    # Phase 2: tau_SVM (Matrix Operation Optimization)
    min_corr = float('inf')
    tau_SVM_sample = 1
    sig_matrix = np.array([signals[ch][start_idx:end_idx] for ch in target_electrodes])
    
    for lag in LAG_SAMPLES:
        sig_t = sig_matrix[:, :-lag] if lag > 0 else sig_matrix
        sig_t_lag = sig_matrix[:, lag:] if lag > 0 else sig_matrix
        sig_t_c = sig_t - np.mean(sig_t, axis=1, keepdims=True)
        sig_t_lag_c = sig_t_lag - np.mean(sig_t_lag, axis=1, keepdims=True)
        C_XY = np.dot(sig_t_c, sig_t_lag_c.T) / sig_t.shape[1]
        
        corr_norm = np.linalg.norm(C_XY, ord='fro')
        if corr_norm < min_corr:
            min_corr = corr_norm
            tau_SVM_sample = lag

    tau_SVM_ms = (tau_SVM_sample / FS) * 1000
    delta_tau = tau_SVM_ms - tau_TE_ms

    return {
        'Time_sec': current_time_sec,
        'Phase': phase_label,
        'Time_to_Seizure': time_to_seizure,
        'Driver_Node': driver_ch,
        'Min_SampEn': round(min_se, 3),
        'tau_TE_ms': round(tau_TE_ms, 2),
        'tau_SVM_ms': round(tau_SVM_ms, 2),
        'Delta_Tau_ms': round(delta_tau, 2)
    }

def process_patient_file(edf_file, seizure_intervals):
    try:
        mne.set_log_level('WARNING')
        raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
        raw.filter(l_freq=30.0, h_freq=80.0, fir_design='firwin', verbose=False)
        
        data, times = raw[:]
        
        available_ch = []
        signals = {}
        for ch_name in raw.ch_names:
            for elec in TARGET_ELECTRODES:
                if elec in ch_name.upper().replace('-', ' '):
                    signals[elec] = data[raw.ch_names.index(ch_name)]
                    if elec not in available_ch: available_ch.append(elec)
                    break
        
        if len(available_ch) < 5: return []

        results = []
        num_samples = data.shape[1]
        
        for (s_start, s_end) in seizure_intervals:
            start_sample = max(0, int((s_start - 900) * FS)) 
            end_sample = min(num_samples - WINDOW_SIZE, int((s_end + 120) * FS)) 
            
            for start_idx in range(start_sample, end_sample, STEP_SIZE):
                res = process_continuous_window(start_idx, data, times, signals, available_ch, seizure_intervals)
                if res:
                    res['File_Name'] = os.path.basename(edf_file)
                    results.append(res)
                
        return results
    except Exception as e:
        return []

if __name__ == "__main__":
    print("=== CHB-MIT Seizure Tracker V3 (Numba Ultra-Fast Edition) ===")
    
    target_edfs = []
    for file_name, intervals in SEIZURE_DICT.items():
        file_path = os.path.join(DATA_DIR, file_name)
        if os.path.exists(file_path):
            target_edfs.append((file_path, intervals))

    print(f"Number of target EDF files: {len(target_edfs)}")
    
    with open(OUTPUT_CSV, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['File_Name', 'Time_sec', 'Phase', 'Time_to_Seizure', 'Driver_Node', 'Min_SampEn', 'tau_TE_ms', 'tau_SVM_ms', 'Delta_Tau_ms'])
        writer.writeheader()

    max_workers = max(1, int(multiprocessing.cpu_count() * 0.8))
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_patient_file, edf, intervals): edf for edf, intervals in target_edfs}
        for i, future in enumerate(as_completed(futures)):
            res_list = future.result()
            if res_list:
                with open(OUTPUT_CSV, mode='a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=res_list[0].keys())
                    writer.writerows(res_list)
            print(f"Progress: {i + 1} / {len(target_edfs)} files completed")

    print("\nContinuous tracking process (V3) completed successfully!")
