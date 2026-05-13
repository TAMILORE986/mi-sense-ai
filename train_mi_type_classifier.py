import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# ── MI type mapping from PTB-XL SCP codes ─────────────────────────────────────
MI_TYPE_MAP = {
    'IMI':   'Inferior MI',
    'IPMI':  'Inferoposterior MI',
    'ILMI':  'Inferolateral MI',
    'IPLMI': 'Inferoposterolateral MI',
    'AMI':   'Anterior MI',
    'ASMI':  'Anteroseptal MI',
    'ALMI':  'Anterolateral MI',
    'LMI':   'Lateral MI',
    'PMI':   'Posterior MI',
}

def extract_mi_type(scp_code_str):
    """
    Reads the scp_codes string from PTB-XL metadata and returns
    the MI type label, or None if the record is not an MI.
    """
    for code, label in MI_TYPE_MAP.items():
        if code in str(scp_code_str):
            return label
    return None


def build_mi_type_dataset(metadata_path, clinical_path):
    print("Loading metadata and clinical data...")
    df           = pd.read_csv(metadata_path, index_col='ecg_id')
    clinical_df  = pd.read_csv(clinical_path, index_col='ecg_id')

    # Keep only MI records
    df['mi_type'] = df['scp_codes'].apply(extract_mi_type)
    mi_df         = df[df['mi_type'].notna()].copy()

    print(f"Total MI records found: {len(mi_df)}")
    print(mi_df['mi_type'].value_counts())

    # Merge with clinical features
    merged = mi_df[['mi_type']].join(clinical_df, how='inner')
    merged = merged.dropna()

    # Drop non-feature columns
    drop_cols = [c for c in ['record_id'] if c in merged.columns]
    feature_cols = [c for c in merged.columns if c != 'mi_type' and c not in drop_cols]

    X = merged[feature_cols]
    y = merged['mi_type']

    return X, y, feature_cols


def train_mi_type_classifier():
    required = ['ptbxl_database.csv', 'clinical_data_full.csv']
    missing  = [f for f in required if not os.path.exists(f)]
    if missing:
        print(f"Error: Missing files: {missing}")
        print("Please ensure simulate_tabular_data_full.py has been run first.")
        return

    X, y, feature_names = build_mi_type_dataset('ptbxl_database.csv', 'clinical_data_full.csv')

    # Encode string labels to integers
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"\nClasses ({len(le.classes_)}): {list(le.classes_)}")

    # Save label encoder mapping
    label_map = {int(i): label for i, label in enumerate(le.classes_)}
    with open('mi_type_label_map.json', 'w') as f:
        json.dump(label_map, f, indent=2)
    print("Label map saved to mi_type_label_map.json")

    # ── Train/test split ──────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # ── Multi-class XGBoost ───────────────────────────────────────────────────
    # Regularisation kept strong to avoid overfitting on smaller per-class samples
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=len(le.classes_),
        eval_metric='mlogloss',
        use_label_encoder=False,
        n_estimators=80,
        learning_rate=0.05,
        max_depth=2,
        subsample=0.7,
        colsample_bytree=0.7,
        gamma=1.0,
        reg_lambda=3.0,
        reg_alpha=0.5,
        min_child_weight=3,
        random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=10
    )

    # ── Evaluation ────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_)

    print(f"\nMI Type Classifier Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    print("\nClassification Report:")
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted', fontsize=11)
    plt.ylabel('Actual', fontsize=11)
    plt.title('Confusion Matrix — MI Type Classifier', fontsize=13)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig('confusion_matrix_mi_type.png', dpi=150)
    plt.close()

    # Save model and feature names
    model.save_model('mi_type_classifier.json')
    with open('mi_type_feature_names.txt', 'w') as f:
        for col in feature_names:
            f.write(f"{col}\n")

    with open('mi_type_performance.txt', 'w') as f:
        f.write(f"MI Type Classifier Accuracy: {acc:.4f}\n\n")
        f.write(report)

    print("\nMI Type Classifier training complete!")
    print("Files saved: mi_type_classifier.json, mi_type_label_map.json,")
    print("             confusion_matrix_mi_type.png, mi_type_performance.txt")


if __name__ == '__main__':
    train_mi_type_classifier()
