
import pandas as pd
import numpy as np
import os
import wfdb
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def preprocess_ecg(signal, fs=100):
    # Denoising: Bandpass filter (0.5 - 40 Hz)
    filtered_signal = butter_bandpass_filter(signal, 0.5, 40.0, fs)
    
    # Normalization: Z-score
    mean = np.mean(filtered_signal, axis=0)
    std = np.std(filtered_signal, axis=0)
    normalized_signal = (filtered_signal - mean) / (std + 1e-8)
    
    # Ensure fixed length (1000 samples for 10s at 100Hz)
    if normalized_signal.shape[0] > 1000:
        normalized_signal = normalized_signal[:1000, :]
    elif normalized_signal.shape[0] < 1000:
        pad_width = ((0, 1000 - normalized_signal.shape[0]), (0, 0))
        normalized_signal = np.pad(normalized_signal, pad_width, mode='constant')
        
    return normalized_signal

def load_and_integrate_data(metadata_path, clinical_path, scp_path, data_dir, output_prefix):
    print(f"Loading metadata and clinical data...")
    df = pd.read_csv(metadata_path, index_col='ecg_id')
    clinical_df = pd.read_csv(clinical_path, index_col='ecg_id')
    scp_df = pd.read_csv(scp_path, index_col=0)
    
    # Extract labels (MI vs non-MI)
    mi_classes = scp_df[scp_df.diagnostic_class == 'MI'].index
    df['label'] = df.scp_codes.apply(lambda x: 1 if any(mi in x for mi in mi_classes) else 0)
    
    ecg_data_list = []
    clinical_data_list = []
    labels_list = []
    
    count = 0
    total = len(df)
    
    print(f"Starting processing for {total} records...")
    for ecg_id, row in df.iterrows():
        try:
            # Load ECG signal
            record_path = os.path.join(data_dir, row['filename_lr'])
            signal, _ = wfdb.rdsamp(record_path)
            
            # Preprocess ECG
            processed_ecg = preprocess_ecg(signal)
            
            # Get clinical features
            clinical_features = clinical_df.loc[ecg_id].values
            
            # Store data
            ecg_data_list.append(processed_ecg)
            clinical_data_list.append(clinical_features)
            labels_list.append(row['label'])
            
            count += 1
            if count % 100 == 0:
                print(f"Processed {count}/{total} records...")
                
        except Exception as e:
            print(f"Error processing record {ecg_id}: {e}")
            continue
            
    # Convert to numpy arrays
    print("Converting to final arrays...")
    X_ecg = np.array(ecg_data_list)
    X_clinical = np.array(clinical_data_list)
    y = np.array(labels_list)
    
    # Save integrated data
    np.save(f"{output_prefix}_ecg_data.npy", X_ecg)
    np.save(f"{output_prefix}_labels.npy", y)
    
    # Save clinical data as CSV for convenience
    integrated_clinical_df = pd.DataFrame(X_clinical, columns=clinical_df.columns)
    integrated_clinical_df.to_csv(f"{output_prefix}_clinical_data.csv", index=False)
    
    print(f"Integration complete. Files saved with prefix: {output_prefix}")

if __name__ == '__main__':
    # Adjust these paths for your local setup
    metadata_file = 'ptbxl_database.csv'
    clinical_file = 'clinical_data_full.csv'
    scp_file = 'scp_statements.csv'
    data_folder = 'ptb-xl' # This should contain the 'records100/' directory
    output_name = 'full_integrated'
    
    if os.path.exists(metadata_file) and os.path.exists(clinical_file) and os.path.exists(scp_file):
        load_and_integrate_data(metadata_file, clinical_file, scp_file, data_folder, output_name)
    else:
        print("Error: Required files not found. Please check metadata, clinical, and scp files.")
