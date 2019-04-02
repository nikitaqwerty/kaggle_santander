"""Catboost 5-folds."""

import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings

from tqdm import tqdm
from sklearn.preprocessing import scale
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

def feature_generator(df, vcs_train_test):
    for i in tqdm(range(200)):
        col = "var_" + str(i)
        vtraintest = vcs_train_test[col]
        t = vtraintest[df[col]].fillna(0).values

        df[col + '_train_test_sum_vcs'] = t
        df[col + '_train_test_sum_vcs_prod'] = df[col] * t
        df[col + '_train_test_sum_vcs_minus'] = scale(df[col]) - scale(t)
        df[col + '_train_test_sum_vcs_plus'] = scale(df[col]) + scale(t)

def main():

    input_path = "../input/"
    output_path = "../output/"

    print("lgb:", lgb.__version__)
    print("loading data...")

    train_df = pd.read_csv(input_path + 'train.csv.zip')

    label = train_df.target
    train = train_df.drop(['ID_code', 'target'], axis=1)

    test = pd.read_csv(input_path + 'test.csv.zip')
    test = test.drop(['ID_code'], axis=1)

    test_filtered = pd.read_pickle(input_path + 'test_filtered.pkl')
    test_filtered = test_filtered.loc[:, train.columns]

    train_test = pd.concat([train, test_filtered]).reset_index(drop=True)

    vcs_train_test = {}

    for col in tqdm(train.columns):
        vcs_train_test[col] = train_test.loc[:, col].value_counts()

    feature_generator(train, vcs_train_test)
    feature_generator(test, vcs_train_test)

    rounds = 10000
    early_stop_rounds = 300

    params = {'lambda_l1': 0,
              'lambda_l2': 3,
              'feature_fraction': 0.9,
              'learning_rate': 0.03,
              'max_depth': 4,
              'boosting_type': 'gbrt',
              'objective': 'binary',
              'metric': 'auc',
              'device': 'gpu',
              'gpu_platform_id': '0',
              'gpu_device_id': '0',
              'max_bin': 255,
              'n_jobs': -1,
              'verbose': -1}

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(train))
    predictions = np.zeros(len(test))

    for train_index, valid_index in skf.split(train, label):

        X_train = train.iloc[train_index]
        X_valid = train.iloc[valid_index]

        y_train = label.iloc[train_index]
        y_valid = label.iloc[valid_index]

        d_train = lgb.Dataset(X_train, y_train)
        d_valid = lgb.Dataset(X_valid, y_valid)

        model = lgb.train(params,
                          d_train,
                          num_boost_round=rounds,
                          valid_sets=[d_train, d_valid],
                          valid_names=['train', 'valid'],
                          early_stopping_rounds=early_stop_rounds,
                          verbose_eval=500)

        oof[val_idx] = model.predict(X_valid)
        predictions += model.predict(test) / skf.n_splits

    auc = round(roc_auc_score(df.label, df["lgb"]), 5)
    print("\nROC AUC:", auc)

    np.save(output_path + f"lgb_{auc}_oof.npy", oof)
    np.save(output_path + f"lgb_{auc}_test.npy", predictions)

if __name__ == "__main__":
    main()
