from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, make_scorer
from skopt import BayesSearchCV
import argparse
import pandas as pd
import numpy as np
import re
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Training pipeline for stacking ensemble classifiers.")
    parser.add_argument("--data_path", type=str, default="../data/MergeTrain.csv",
                        help="Path to the training data CSV file.")
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
    return train_test_split(X, y, test_size=0.2, random_state=42)


# 假设已加载数据和特征

args = parse_args()
final_features = ['stock_entry_mean', 'is_virtual', 'online_duration', 'is_online_duration_less_than_3_month',
                  'dur_day_rate', 'is_cross_city_src', 'is_cross_province_src', 'is_not_in_home_src',
                  'city_count_src', 'province_count_src', 'phone2opposite_skew', 'call_day_count_src',
                  'call_day_count_dst', 'caller_external_province_discreteness_non_std', 'src_rate_non_std',
                  'total_call_count', 'total_call_duration_non_std', 'user_count', 'city_count', 'province_count',
                  'phone2opposite_mean_non_std', 'phone2opposite_median_non_std', 'phone2opposite_max',
                  'phone2opposite_sem_non_std', 'phone2opposite_skew_non_std', 'hour_mean_non_std', 'hour_std',
                  'hour_skew_non_std', 'hour_sem_non_std', 'week_mean_non_std', 'week_std_non_std',
                  'week_skew_non_std', 'week_sem_non_std', 'month_mean_non_std', 'month_std_non_std',
                  'day_count_mean_non_std', 'day_count_std', 'day_count_max', 'day_count_skew',
                  'day_count_sem_non_std', 'avg_call_duration', 'interval_mean', 'interval_min', 'interval_std',
                  'interval_skew_non_std', 'interval_sem_non_std', 'cfee_mean', 'cfee_std', 'cfee_total_non_std',
                  'cfee_sem', 'cfee_skew', 'lfee_mean', 'lfee_std', 'lfee_total', 'lfee_sem', 'lfee_skew',
                  'stock_entry_std_non_std', 'stock_entry_skew', 'stock_entry_sem', 'entry_delay_mean',
                  'entry_delay_std', 'entry_delay_skew_non_std', 'entry_delay_sem', 'call_duration_mean',
                  'call_duration_std_non_std', 'call_duration_skew_non_std', 'call_duration_sem', 'call_day_count',
                  'day_call_max_div_call_day', 'src_dst_rate', 'dst_count', 'is_dst_count_less_than_20',
                  'is_src_std_non_std', 'src_user_count', 'dst_user_count', 'total_count', 'src_user_rate',
                  'caller_opponent_discreteness_non_std', 'roam_src_count_non_std', 'roam_src_rate', 'is_vip_mean',
                  'silence_and_high_freq', 'home_location_call_count', 'number_lifespan', 'active_streak',
                  'call_cycle_variance']
# 定义F1评分函数
f1_scorer = make_scorer(f1_score, average='binary')

# Step 1: 使用网格搜索粗略找到参数范围
param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'num_leaves': [20, 40, 60],
    'learning_rate': [0.05, 0.1, 0.15],
    'subsample': [0.6, 0.8, 1.0],
    'class_weight': ['balanced']  # 处理不平衡
}

grid_search = GridSearchCV(
    estimator=LGBMClassifier(random_state=42),
    param_grid=param_grid,
    scoring=f1_scorer,
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
print("Grid Search Best Params:", grid_search.best_params_)
print("Grid Search Best F1 Score:", grid_search.best_score_)

# 根据网格搜索确定的最佳参数范围缩小贝叶斯优化的范围
bayes_params = {
    'n_estimators': (grid_search.best_params_['n_estimators'] - 50, grid_search.best_params_['n_estimators'] + 50),
    'num_leaves': (max(15, grid_search.best_params_['num_leaves'] - 20), grid_search.best_params_['num_leaves'] + 20),
    'learning_rate': (0.01, grid_search.best_params_['learning_rate'] + 0.05),
    'subsample': (0.5, 1.0)
}

# Step 2: 使用贝叶斯优化进一步细化参数
bayes_search = BayesSearchCV(
    estimator=LGBMClassifier(random_state=42, class_weight='balanced'),
    search_spaces=bayes_params,
    scoring=f1_scorer,
    cv=5,
    n_iter=30,  # 贝叶斯优化的迭代次数，适当增加可以提高精度
    n_jobs=-1,
    verbose=1,
    random_state=42
)

# 执行贝叶斯搜索
bayes_search.fit(X_train, y_train)
print("Bayesian Optimization Best Params:", bayes_search.best_params_)
print("Bayesian Optimization Best F1 Score:", bayes_search.best_score_)

# Step 3: 用找到的最佳参数训练最终模型并评估
best_lgb = bayes_search.best_estimator_
y_pred = best_lgb.predict(X_test)
f1 = f1_score(y_test, y_pred)
print("Test F1 Score:", f1)
