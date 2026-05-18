import gdown, os

files = {
    "mi_detection_model_full.h5": "YOUR_GDRIVE_FILE_ID_1",
    "xgboost_model.json":         "YOUR_GDRIVE_FILE_ID_2",
    "mi_type_classifier.json":    "YOUR_GDRIVE_FILE_ID_3",
    "mi_type_label_map.json":     "YOUR_GDRIVE_FILE_ID_4",
    "xgboost_feature_names.txt":  "YOUR_GDRIVE_FILE_ID_5",
}

for filename, file_id in files.items():
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}",
                       filename, quiet=False)
    else:
        print(f"{filename} already exists.")