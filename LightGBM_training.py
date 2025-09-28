import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    fbeta_score,
    precision_recall_curve,
    average_precision_score,
    balanced_accuracy_score,
)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import warnings
import sys
from datetime import datetime
import os

warnings.filterwarnings("ignore")


def calculate_scale_pos_weight(y, cost_ratio=10):
    """
    Calculate scale_pos_weight for LightGBM cost-sensitive learning.
    
    Args:
        y: Target variable
        cost_ratio: How much more costly false negatives are compared to false positives
                   (e.g., 10 means missing a death is 10x worse than a false alarm)
    
    Returns:
        Float value for LightGBM's scale_pos_weight parameter
    """
    class_counts = np.bincount(y)
    n_neg = class_counts[0]  # No death count
    n_pos = class_counts[1]  # Death count
    
    # Basic class imbalance ratio
    base_ratio = n_neg / n_pos
    
    # Apply cost ratio to emphasize positive class even more
    scale_pos_weight = base_ratio * cost_ratio
    
    return scale_pos_weight


def calculate_class_weights_info(y, cost_ratio=10):
    """
    Calculate class weight information for display purposes.
    
    Returns:
        Dictionary with class weight info for logging
    """
    scale_pos_weight = calculate_scale_pos_weight(y, cost_ratio)
    class_counts = np.bincount(y)
    
    return {
        "scale_pos_weight": scale_pos_weight,
        "no_death_count": class_counts[0],
        "death_count": class_counts[1],
        "base_ratio": class_counts[0] / class_counts[1],
        "cost_ratio": cost_ratio
    }


def calculate_f2_score(y_true, y_pred_proba, threshold=0.5):
    """
    Calculate F2-score (recall weighted 2x more than precision)
    """
    y_pred = (y_pred_proba > threshold).astype(int)
    return fbeta_score(y_true, y_pred, beta=2)


# Output redirection class
class OutputLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()


def load_and_preprocess_data():
    """Load and preprocess the COVID-19 data for modeling"""
    print("Loading data...")
    df = pd.read_csv("cases.csv", low_memory=False)

    print(f"Original data shape: {df.shape}")
    print(f"Missing values per column:\n{df.isnull().sum()}")

    # Remove rows where death status is unknown (our target variable)
    df_clean = df.dropna(subset=["death"]).copy()
    print(f"After removing unknown death status: {df_clean.shape}")

    # Remove rows where all features are missing
    feature_cols = ["sex", "age_group", "race", "hosp", "icu"]
    df_clean = df_clean.dropna(subset=feature_cols)
    print(f"After removing rows with missing features: {df_clean.shape}")

    return df_clean


def create_features(df):
    """Create engineered features for better model performance"""
    df_featured = df.copy()

    # Create age group numeric mapping (higher number = higher risk)
    age_mapping = {
        "0 - 9 Years": 1,
        "10 - 19 Years": 2,
        "20 - 29 Years": 3,
        "30 - 39 Years": 4,
        "40 - 49 Years": 5,
        "50 - 59 Years": 6,
        "60 - 69 Years": 7,
        "70 - 79 Years": 8,
        "80+ Years": 9,
    }
    df_featured["age_numeric"] = df_featured["age_group"].map(age_mapping)

    # Create high-risk age groups
    df_featured["high_risk_age"] = (df_featured["age_numeric"] >= 7).astype(int)
    df_featured["very_high_risk_age"] = (df_featured["age_numeric"] >= 8).astype(int)

    # Create risk combinations
    df_featured["hosp_icu_risk"] = (df_featured["hosp"] == "Yes").astype(int) + (
        df_featured["icu"] == "Yes"
    ).astype(int)

    return df_featured


def prepare_model_data(df):
    """Prepare data for LightGBM training"""
    # Select features for modeling
    feature_cols = [
        "age_numeric",
        "high_risk_age",
        "very_high_risk_age",
        "hosp_icu_risk",
    ]

    # Encode categorical variables appropriately
    df_encoded = df.copy()

    # 1. Age group: Use LabelEncoder (has natural order)
    le_age = LabelEncoder()
    df_encoded["age_group_encoded"] = le_age.fit_transform(
        df_encoded["age_group"].fillna("Unknown")
    )

    # 2. Sex: Use One-Hot Encoding (nominal, no order)
    sex_dummies = pd.get_dummies(df_encoded["sex"].fillna("Unknown"), prefix="sex")
    df_encoded = pd.concat([df_encoded, sex_dummies], axis=1)

    # 3. Race: Use One-Hot Encoding (nominal, no order)
    race_dummies = pd.get_dummies(df_encoded["race"].fillna("Unknown"), prefix="race")
    df_encoded = pd.concat([df_encoded, race_dummies], axis=1)

    # 4. Hospitalization: Binary encoding (Yes/No/Unknown)
    df_encoded["hosp_yes"] = (df_encoded["hosp"] == "Yes").astype(int)
    df_encoded["hosp_unknown"] = df_encoded["hosp"].isna().astype(int)

    # 5. ICU: Binary encoding (Yes/No/Unknown)
    df_encoded["icu_yes"] = (df_encoded["icu"] == "Yes").astype(int)
    df_encoded["icu_unknown"] = df_encoded["icu"].isna().astype(int)

    # Add encoded categorical features to feature list
    categorical_features = (
        ["age_group_encoded"]
        + list(sex_dummies.columns)
        + list(race_dummies.columns)
        + ["hosp_yes", "hosp_unknown", "icu_yes", "icu_unknown"]
    )
    feature_cols.extend(categorical_features)

    X = df_encoded[feature_cols]
    y = (df_encoded["death"] == "Yes").astype(int)

    # Clean feature names to remove special characters that LightGBM doesn't support
    clean_feature_names = []
    for col in X.columns:
        # Replace problematic characters with underscores
        clean_name = (col.replace("/", "_")
                        .replace(",", "")
                        .replace(" ", "_")
                        .replace("(", "")
                        .replace(")", "")
                        .replace("-", "_")
                        .replace("'", ""))
        # Remove multiple consecutive underscores
        while "__" in clean_name:
            clean_name = clean_name.replace("__", "_")
        # Remove leading/trailing underscores
        clean_name = clean_name.strip("_")
        clean_feature_names.append(clean_name)
    
    # Rename columns to clean names
    X.columns = clean_feature_names
    
    return X, y, clean_feature_names


def train_lightgbm_model(X, y, best_params=None, test_size=0.2, random_state=42, cost_ratio=10):
    """Train LightGBM model with cost-sensitive learning"""
    print("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Death rate in training set: {y_train.mean():.4f}")
    print(f"Death rate in test set: {y_test.mean():.4f}")
    
    # Calculate cost-sensitive scale_pos_weight
    scale_pos_weight = calculate_scale_pos_weight(y_train, cost_ratio=cost_ratio)
    weight_info = calculate_class_weights_info(y_train, cost_ratio=cost_ratio)
    print(f"Cost-sensitive learning: scale_pos_weight = {scale_pos_weight:.2f}")
    print(f"Base class ratio (No Death:Death) = {weight_info['base_ratio']:.1f}:1")
    print(f"With cost ratio {cost_ratio}:1, effective weight = {scale_pos_weight:.1f}x higher for Death class")

    # Use optimized parameters if provided, otherwise use default
    if best_params is not None:
        lgb_params = best_params.copy()
        lgb_params["verbose"] = -1
        lgb_params["random_state"] = random_state
        # Override with cost-sensitive scale_pos_weight
        lgb_params["scale_pos_weight"] = scale_pos_weight
        print("Using Optuna-optimized parameters with cost-sensitive weights...")
    else:
        # Default LightGBM parameters for cost-sensitive learning
        lgb_params = {
            "objective": "binary",
            "metric": ["auc", "binary_logloss"],  # Keep AUC for comparison but optimize for F2
            "boosting_type": "gbdt", 
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "random_state": random_state,
            "scale_pos_weight": scale_pos_weight,
        }
        print("Using default parameters with cost-sensitive weights...")

    print("Training LightGBM model...")
    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    return model, X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test, feature_names):
    """Evaluate model performance with enhanced metrics for imbalanced data"""
    print("\n" + "=" * 60)
    print("MODEL EVALUATION - COST-SENSITIVE LEARNING")
    print("=" * 60)

    # Make predictions
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Class distribution analysis
    death_rate = y_test.mean()
    print(f"Class Distribution in Test Set:")
    print(f"  No Death: {(1-death_rate)*100:.2f}% ({(y_test == 0).sum():,} samples)")
    print(f"  Death: {death_rate*100:.2f}% ({(y_test == 1).sum():,} samples)")
    print(f"  Class Imbalance Ratio: {(1-death_rate)/death_rate:.1f}:1 (No Death:Death)")

    # Calculate comprehensive metrics
    auc_score = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    f1_score = fbeta_score(y_test, y_pred, beta=1)
    f2_score = fbeta_score(y_test, y_pred, beta=2)
    f05_score = fbeta_score(y_test, y_pred, beta=0.5)

    print(f"\n📊 COMPREHENSIVE METRICS:")
    print(f"AUC-ROC: {auc_score:.4f} (traditional metric)")
    print(f"PR-AUC: {pr_auc:.4f} (better for imbalanced data)")
    print(f"Balanced Accuracy: {balanced_acc:.4f} (accounts for class imbalance)")
    print(f"F1-score: {f1_score:.4f} (balanced precision/recall)")
    print(f"F2-score: {f2_score:.4f} ⭐ (emphasizes recall - our target metric)")
    print(f"F0.5-score: {f05_score:.4f} (emphasizes precision)")

    # Classification report
    print(f"\n📋 CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, target_names=["No Death", "Death"]))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"True Negatives: {cm[0, 0]:,}")
    print(f"False Positives: {cm[0, 1]:,}")
    print(f"False Negatives: {cm[1, 0]:,}")
    print(f"True Positives: {cm[1, 1]:,}")

    # Feature importance
    print(f"\nFeature Importance:")
    feature_importance = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importance()}
    ).sort_values("importance", ascending=False)

    print(feature_importance)

    return y_pred_proba, y_pred, feature_importance


def plot_results(y_test, y_pred_proba, feature_importance, output_dir="plots"):
    """Create visualization plots focused on cost-sensitive learning metrics"""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Saving plots to '{output_dir}' directory...")
    
    # Calculate key metrics
    y_pred = (y_pred_proba > 0.5).astype(int)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    f2_score = fbeta_score(y_test, y_pred, beta=2)
    
    # 1. Precision-Recall Curve (Better for Imbalanced Data)
    plt.figure(figsize=(8, 6))
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    
    plt.plot(recall, precision, color="darkorange", lw=2, label=f"PR curve (AUC = {pr_auc:.3f})")
    plt.axhline(y=y_test.mean(), color="navy", lw=2, linestyle="--", label=f"Random (AUC = {y_test.mean():.3f})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision (PPV)")
    plt.title("Precision-Recall Curve\n(Better Metric for Imbalanced Data)")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.text(0.6, 0.2, f"F2-Score: {f2_score:.3f}", fontsize=12, bbox=dict(boxstyle="round", facecolor="wheat"))
    plt.tight_layout()
    plt.savefig(f"{output_dir}/precision_recall_curve.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # 2. ROC Curve (Keep for comparison)
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc_score:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve\n(Traditional Metric - Less Reliable for Imbalanced Data)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # 2. Feature Importance
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(15)
    plt.barh(range(len(top_features)), top_features["importance"])
    plt.yticks(range(len(top_features)), top_features["feature"])
    plt.xlabel("Importance")
    plt.title("Top 15 Feature Importance")
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # 3. Prediction Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(y_pred_proba[y_test == 0], bins=50, alpha=0.7, label="No Death", color="blue")
    plt.hist(y_pred_proba[y_test == 1], bins=50, alpha=0.7, label="Death", color="red")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.title("Prediction Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/prediction_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # 4. Confusion Matrix Heatmap
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, (y_pred_proba > 0.5).astype(int))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["No Death", "Death"], 
                yticklabels=["No Death", "Death"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # 5. Cost-Sensitive Metrics Dashboard
    plt.figure(figsize=(12, 8))
    
    # Calculate various F-beta scores and other metrics
    f1_score = fbeta_score(y_test, y_pred, beta=1)
    f05_score = fbeta_score(y_test, y_pred, beta=0.5)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    
    # Metrics comparison
    metrics = ['PR-AUC', 'ROC-AUC', 'F2-Score', 'F1-Score', 'F0.5-Score', 'Balanced\nAccuracy']
    values = [pr_auc, auc_score, f2_score, f1_score, f05_score, balanced_acc]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    bars = plt.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    plt.ylim(0, 1)
    plt.ylabel('Score')
    plt.title('Cost-Sensitive Learning: Comprehensive Metrics Dashboard\n(F2-Score Optimized Model)', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Highlight F2-score as our target metric
    bars[2].set_color('#FF1744')  # Make F2-score bar red to highlight
    plt.text(2, f2_score + 0.05, '⭐ TARGET METRIC', ha='center', va='bottom',
             fontsize=10, fontweight='bold', color='#FF1744')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cost_sensitive_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # 6. Combined results plot (updated 2x3 grid)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Cost-Sensitive Learning Results Dashboard (F2-Score Optimized)', 
                 fontsize=16, fontweight='bold')
    
    # Precision-Recall Curve (subplot)
    axes[0, 0].plot(recall, precision, color="darkorange", lw=2, label=f"PR-AUC = {pr_auc:.3f}")
    axes[0, 0].axhline(y=y_test.mean(), color="navy", lw=2, linestyle="--", label="Random")
    axes[0, 0].set_xlim([0.0, 1.0])
    axes[0, 0].set_ylim([0.0, 1.05])
    axes[0, 0].set_xlabel("Recall (Sensitivity)")
    axes[0, 0].set_ylabel("Precision (PPV)")
    axes[0, 0].set_title("⭐ Precision-Recall Curve")
    axes[0, 0].legend(loc="lower left")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Feature Importance (subplot)
    top_features_sub = feature_importance.head(10)
    axes[0, 1].barh(range(len(top_features_sub)), top_features_sub["importance"])
    axes[0, 1].set_yticks(range(len(top_features_sub)))
    axes[0, 1].set_yticklabels(top_features_sub["feature"])
    axes[0, 1].set_xlabel("Importance")
    axes[0, 1].set_title("Top 10 Feature Importance")
    axes[0, 1].invert_yaxis()
    axes[0, 1].grid(True, alpha=0.3, axis='x')
    
    # Metrics Comparison (subplot)
    metrics_short = ['PR-AUC', 'ROC-AUC', 'F2', 'F1', 'F0.5', 'Bal-Acc']
    axes[0, 2].bar(metrics_short, values, color=colors, alpha=0.8)
    axes[0, 2].set_ylim(0, 1)
    axes[0, 2].set_ylabel('Score')
    axes[0, 2].set_title('⭐ Metrics Comparison')
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # Add F2 highlight
    axes[0, 2].bar(2, f2_score, color='#FF1744', alpha=0.9)
    
    # Prediction Distribution (subplot)
    axes[1, 0].hist(y_pred_proba[y_test == 0], bins=50, alpha=0.7, label="No Death", color="blue")
    axes[1, 0].hist(y_pred_proba[y_test == 1], bins=50, alpha=0.7, label="Death", color="red")
    axes[1, 0].set_xlabel("Predicted Probability")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title("Prediction Distribution")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Confusion Matrix Heatmap (subplot)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1, 1],
                xticklabels=["No Death", "Death"], 
                yticklabels=["No Death", "Death"])
    axes[1, 1].set_xlabel("Predicted")
    axes[1, 1].set_ylabel("Actual")
    axes[1, 1].set_title("Confusion Matrix")
    
    # ROC Curve (subplot) - kept for comparison
    axes[1, 2].plot(fpr, tpr, color="orange", lw=2, label=f"ROC-AUC = {auc_score:.3f}")
    axes[1, 2].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    axes[1, 2].set_xlim([0.0, 1.0])
    axes[1, 2].set_ylim([0.0, 1.05])
    axes[1, 2].set_xlabel("False Positive Rate")
    axes[1, 2].set_ylabel("True Positive Rate")
    axes[1, 2].set_title("ROC Curve (Reference)")
    axes[1, 2].legend(loc="lower right")
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/combined_results.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"All plots saved successfully in '{output_dir}' directory!")


def optimize_hyperparameters(X, y, n_trials=100, cv_folds=5, cost_ratio=10):
    """Use Optuna to optimize LightGBM hyperparameters for F2-score"""
    print(f"\nStarting Optuna optimization with {n_trials} trials...")
    print("Optimizing for F2-score (prioritizes recall over precision)")

    def objective(trial):
        # Define hyperparameter search space
        params = {
            "objective": "binary",
            "metric": ["auc", "binary_logloss"],  # Keep multiple metrics for monitoring
            "boosting_type": "gbdt",
            "num_leaves": trial.suggest_int("num_leaves", 10, 100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "verbose": -1,
            "random_state": 42,
        }

        # Cross-validation with F2-score optimization
        f2_scores = []
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        for train_idx, val_idx in skf.split(X, y):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # Calculate cost-sensitive scale_pos_weight for this fold
            fold_scale_pos_weight = calculate_scale_pos_weight(y_train_fold, cost_ratio=cost_ratio)
            params["scale_pos_weight"] = fold_scale_pos_weight

            train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
            model = lgb.train(
                params,
                train_data,
                num_boost_round=500,
                valid_sets=[train_data],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
            )

            # Calculate F2-score on validation fold
            y_pred_proba_fold = model.predict(X_val_fold)
            f2_fold = calculate_f2_score(y_val_fold, y_pred_proba_fold)
            f2_scores.append(f2_fold)

        return np.mean(f2_scores)

    # Create study and optimize
    study = optuna.create_study(
        direction="maximize",
        study_name="lightgbm_covid_f2_optimization",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20),
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nOptuna optimization completed!")
    print(f"Best F2-score: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")

    return study.best_params


def cross_validate_model(X, y, cv_folds=5, cost_ratio=10):
    """Perform cross-validation with cost-sensitive learning and F2-score"""
    print(f"\nPerforming {cv_folds}-fold cross-validation with cost-sensitive learning...")

    # Calculate scale_pos_weight for cost-sensitive learning
    scale_pos_weight = calculate_scale_pos_weight(y, cost_ratio=cost_ratio)
    weight_info = calculate_class_weights_info(y, cost_ratio=cost_ratio)
    print(f"Using scale_pos_weight = {scale_pos_weight:.2f} (base ratio: {weight_info['base_ratio']:.1f}:1, cost ratio: {cost_ratio}:1)")

    lgb_params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "scale_pos_weight": scale_pos_weight,
    }

    auc_scores = []
    f2_scores = []
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
        model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=500,
            valid_sets=[train_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )

        y_pred_fold = model.predict(X_val_fold)
        auc_fold = roc_auc_score(y_val_fold, y_pred_fold)
        f2_fold = calculate_f2_score(y_val_fold, y_pred_fold)
        
        auc_scores.append(auc_fold)
        f2_scores.append(f2_fold)
        
        print(f"Fold {fold + 1} - AUC: {auc_fold:.4f}, F2: {f2_fold:.4f}")

    print(f"\nCross-validation results:")
    print(f"Mean AUC: {np.mean(auc_scores):.4f} (+/- {np.std(auc_scores) * 2:.4f})")
    print(f"Mean F2-score: {np.mean(f2_scores):.4f} (+/- {np.std(f2_scores) * 2:.4f})")
    print(f"Individual AUC scores: {[f'{score:.4f}' for score in auc_scores]}")
    print(f"Individual F2 scores: {[f'{score:.4f}' for score in f2_scores]}")

    return auc_scores, f2_scores


def main(use_optuna=True, n_trials=100, cost_ratio=10):
    """Main function to run the complete cost-sensitive learning pipeline"""
    # Set up output logging with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"training_output_{timestamp}.txt"
    
    # Redirect output to both console and file
    logger = OutputLogger(output_filename)
    sys.stdout = logger
    
    try:
        print("=" * 80)
        print("COVID-19 DEATH PREDICTION - COST-SENSITIVE LEARNING")
        print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Cost ratio (Death:No Death penalty): {cost_ratio}:1")
        print(f"Optimization metric: F2-score (emphasizes recall)")
        print(f"Output being saved to: {output_filename}")
        print("=" * 80)

        # Load and preprocess data
        df = load_and_preprocess_data()

        # Create features
        df_featured = create_features(df)
        print(f"After feature engineering: {df_featured.shape}")

        # Prepare model data
        X, y, feature_names = prepare_model_data(df_featured)
        print(f"Final feature matrix shape: {X.shape}")
        print(f"Features: {feature_names}")

        # Hyperparameter optimization with Optuna (F2-score)
        best_params = None
        if use_optuna:
            best_params = optimize_hyperparameters(X, y, n_trials=n_trials, cost_ratio=cost_ratio)
        else:
            print("Skipping Optuna optimization, using default cost-sensitive parameters...")

        # Cross-validation with cost-sensitive learning
        auc_scores, f2_scores = cross_validate_model(X, y, cost_ratio=cost_ratio)

        # Train final model with cost-sensitive learning
        model, X_train, X_test, y_train, y_test = train_lightgbm_model(X, y, best_params, cost_ratio=cost_ratio)

        # Evaluate model
        y_pred_proba, y_pred, feature_importance = evaluate_model(
            model, X_test, y_test, feature_names
        )

        # Create plots directory with timestamp
        plots_dir = f"plots_{timestamp}"
        
        # Plot results
        plot_results(y_test, y_pred_proba, feature_importance, plots_dir)

        # Save model with timestamp
        model_filename = f"covid_death_prediction_model_{timestamp}.txt"
        model.save_model(model_filename)
        print(f"\nModel saved as '{model_filename}'")

        # Save feature importance with timestamp
        feature_importance_filename = f"feature_importance_{timestamp}.csv"
        feature_importance.to_csv(feature_importance_filename, index=False)
        print(f"Feature importance saved as '{feature_importance_filename}'")

        # Save best parameters if using Optuna
        if best_params is not None:
            import json
            
            params_filename = f"best_parameters_{timestamp}.json"
            with open(params_filename, "w") as f:
                json.dump(best_params, f, indent=2)
            print(f"Best parameters saved as '{params_filename}'")

        print("\n" + "=" * 70)
        print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"All outputs saved with timestamp: {timestamp}")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nError occurred during training: {str(e)}")
        raise
    finally:
        # Restore stdout and close log file
        sys.stdout = logger.terminal
        logger.close()
        print(f"\nTraining log saved to: {output_filename}")


if __name__ == "__main__":
    main()
