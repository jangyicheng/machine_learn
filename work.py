import re
import pandas as pd
import numpy as np
import argparse
import sys
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from mlxtend.classifier import StackingClassifier


# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Training pipeline for stacking ensemble classifiers.")
    parser.add_argument("--data_path", type=str, default="../data/train_with_normalize_unique.csv",
                        help="Path to the training data CSV file.")
    parser.add_argument("--run", type=int, default=0, help="Flag to enable hyperparameter tuning.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for prediction.")
    parser.add_argument("--output_path", type=str, default="../commit/submission.csv",
                        help="Path to save prediction results.")
    parser.add_argument("--test_path", type=str, default="../data/valid_with_normalize_unique.csv",
                        help="Path to save prediction results.")
    parser.add_argument("--base_count", type=int, default=5, help="Number of base classifiers for each model.")
    return parser.parse_args()


# Clean column names
def clean_feature_names(df):
    df.columns = [re.sub(r'\W+', '_', str(col)) for col in df.columns]
    return df


# Load, preprocess, and resample data
def load_and_preprocess_data(data_path, final_features):
    data = pd.read_csv(data_path, engine='python')
    data = clean_feature_names(data).round(6)  # Adjust precision
    X, y = data[final_features], data['is_sa']

    # Standardize features
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)

    sys.stdout.write("Data loaded and preprocessed!\n")
    return train_test_split(X, y, test_size=0.1, random_state=42)


# Grid search with early stopping
def perform_grid_search(model, param_grid, X, y):
    grid_search = GridSearchCV(
        estimator=model, param_grid=param_grid, cv=6, n_jobs=10,
        scoring='f1', verbose=1
    )
    grid_search.fit(X, y)
    sys.stdout.write("Grid search completed!\n")
    return grid_search.best_estimator_


# Initialize classifiers with parameter grids
def get_base_classifiers(X_train, y_train, run_flag, base_count):
    # Set up hyperparameter grids with imbalance handling parameters
    param_grids = {
        'XGB': {'n_estimators': [243], 'max_depth': [7], 'learning_rate': [0.0469], 'subsample': [0.835]},
        # 'LGB': {'n_estimators': [342], 'num_leaves': [44], 'learning_rate': [0.05]},
        # 'CAT': {'iterations': [300], 'depth': [7], 'learning_rate': [0.1]}
    }

    # Count instances for calculating scale_pos_weight
    pos_weight = sum(y_train == 0) / sum(y_train == 1)
    pos_weight = 8

    base_classifiers = []
    for name, param_grid in param_grids.items():
        if name == 'XGB':
            model_class = XGBClassifier
            imbalance_params = {'scale_pos_weight': pos_weight}
        elif name == 'LGB':
            model_class = LGBMClassifier
            imbalance_params = {'is_unbalance': True}
        elif name == 'CAT':
            model_class = CatBoostClassifier
            imbalance_params = {'class_weights': [1, pos_weight]}

        for i in range(base_count):
            # Initialize model with imbalance handling parameters
            model = model_class(
                **imbalance_params, verbose=-1 if name != 'CAT' else 0, random_state=42 + i
            )
            if run_flag:
                model = perform_grid_search(model, param_grid, X_train, y_train)
            model.fit(X_train, y_train)
            base_classifiers.append(model)
            print(f"Trained {name} model {i + 1}/{base_count}")

    sys.stdout.write("Base classifiers trained with imbalance handling!\n")
    return base_classifiers


# Stack and train ensemble classifier
def stack_train(base_classifiers, X_train, X_test, y_train, y_test, meta_classifier, threshold):
    stack = StackingClassifier(
        classifiers=base_classifiers,
        meta_classifier=meta_classifier,
        use_features_in_secondary=True
    )
    stack.fit(X_train, y_train)
    y_pred = (stack.predict_proba(X_test)[:, 1] >= threshold).astype(int)

    sys.stdout.write(classification_report(y_test, y_pred))
    sys.stdout.write(f"Accuracy: {accuracy_score(y_test, y_pred)}\n")
    sys.stdout.write(f"F1 Score: {f1_score(y_test, y_pred)}\n")
    sys.stdout.write("Stack training completed!\n")
    return stack


# Predict on test data and save results
def predict_and_save(stack, test_data_path, final_features, threshold, output_path):
    test_data = pd.read_csv(test_data_path)
    test_data = clean_feature_names(test_data).round(6)  # Adjust precision
    X_test_final = test_data[final_features]
    test_data['is_sa'] = (stack.predict_proba(X_test_final)[:, 1] >= threshold).astype(int)
    test_data[['msisdn', 'is_sa']].to_csv(output_path, index=False)
    sys.stdout.write(f"Results saved to {output_path}\n")


def main():
    args = parse_args()
    final_features = ['cfee', 'lfee', 'open_datetime', 'long_type1', 'roam_type', 'month', 'dayofweek', 'hour',
                      'visit_area_code', 'called_code', 'is_virtual', 'is_cross_city', 'is_cross_province',
                      'is_not_in_home', 'stock_entry_time', 'stock_entry_delay', 'is_vip', 'online_duration',
                      'is_online_duration_less_than_3_month', 'is_occur_dur_day', 'dur_day_rate', 'is_cross_city_src',
                      'is_cross_province_src', 'is_not_in_home_src', 'city_count_src', 'province_count_src',
                      'phone2opposite_skew_src', 'call_day_count_src', 'call_day_count_dst',
                      'caller_external_province_discreteness', 'src_rate', 'total_call_count', 'total_call_duration',
                      'user_count', 'city_count', 'province_count', 'phone2opposite_mean', 'phone2opposite_median',
                      'phone2opposite_max', 'phone2opposite_sem', 'phone2opposite_skew', 'hour_mean', 'hour_std',
                      'hour_skew', 'hour_sem', 'week_mean', 'week_std', 'week_skew', 'week_sem', 'month_mean',
                      'month_std', 'day_count_mean', 'day_count_std', 'day_count_max', 'day_count_skew',
                      'day_count_sem', 'avg_call_duration', 'interval_mean', 'interval_min', 'interval_std',
                      'interval_skew', 'interval_sem', 'cfee_mean', 'cfee_std', 'cfee_total', 'cfee_sem', 'cfee_skew',
                      'lfee_mean', 'lfee_std', 'lfee_total', 'lfee_sem', 'lfee_skew', 'stock_entry_mean',
                      'stock_entry_std', 'stock_entry_skew', 'stock_entry_sem', 'entry_delay_mean', 'entry_delay_std',
                      'entry_delay_skew', 'entry_delay_sem', 'call_duration_mean', 'call_duration_std',
                      'call_duration_skew', 'call_duration_sem', 'call_day_count', 'day_call_max_div_call_day',
                      'src_dst_rate', 'dst_count', 'is_dst_count_less_than_20', 'is_src_std', 'src_user_count',
                      'dst_user_count', 'total_count', 'src_user_rate', 'caller_opponent_discreteness',
                      'roam_src_count', 'roam_src_rate', 'is_vip_mean', 'silence_and_high_freq',
                      'home_location_call_count', 'number_lifespan', 'active_streak', 'call_cycle_variance']
    X_train, X_test, y_train, y_test = load_and_preprocess_data(args.data_path, final_features)
    base_classifiers = get_base_classifiers(X_train, y_train, args.run, args.base_count)
    meta_classifier = BalancedRandomForestClassifier(random_state=42)  # Balanced meta-classifier
    stack = stack_train(base_classifiers, X_train, X_test, y_train, y_test, meta_classifier, args.threshold)
    predict_and_save(stack, args.test_path, final_features, args.threshold, args.output_path)


if __name__ == "__main__":
    main()
