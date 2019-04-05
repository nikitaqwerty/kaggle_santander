"""Catboost 5-folds."""

import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings

from tqdm import tqdm
from sklearn.preprocessing import scale
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, ParameterSampler

warnings.filterwarnings("ignore")

def feature_generator(df, vcs_train_test):
    for i in tqdm(range(200)):
        col = "var_" + str(i)
        vtraintest = vcs_train_test[col]
        t = vtraintest[df[col]].fillna(0).values

        df[col + '_train_test_sum_vcs'] = t
        df[col + '_train_test_sum_vcs_prod'] = df[col] * t
        df[col + '_train_test_sum_vcs_unique'] = np.array(t == 1).astype(int)
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

    param_grid = {'num_leaves': list(range(10, 33)),
                  'max_bin': [511, 1023, 2047],
                  'min_data_in_leaf': list(range(10, 100, 10)),
                  'min_sum_hessian_in_leaf': list(range(0, 20)),
                  'learning_rate': np.linspace(0.01, 0.03, 31),
                  "bagging_freq": list(range(1, 10)),
                  "bagging_fraction": np.linspace(0.3, 1.0, 61),
                  'feature_fraction': np.linspace(0.01, 1.0, 101),
                  'lambda_l1': np.linspace(0, 5, 61),
                  'lambda_l2': np.linspace(0, 5, 61)}

    param_static = {'boosting_type': 'gbrt',
                    'objective': 'binary',
                    'tree_learner': 'serial',
                    'metric': 'auc',
                    'bagging_seed': 42,
                    'seed': 42,
                    'max_depth': -1,
                    'verbose': -1,
                    'n_jobs': -1}

    param_list = list(ParameterSampler(param_grid, n_iter=100))

    rounded_list = [dict((k, round(v, 6)) for (k, v) in d.items())
                    for d in param_list]

    results = []
    for i, params in enumerate(rounded_list):
        print(params)

        for key in param_static:
            params[key] = param_static[key]

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        auc = 0

        j = 0
        for train_index, valid_index in skf.split(train, label):
            j += 1

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
                              verbose_eval=0)

            pred = model.predict(X_valid)

            auc += round(roc_auc_score(y_valid, pred), 5)
            auc /= j
            print(i, j, "CV score: {:<8.5f}".format(auc))

            if j >= 2:
                break

        results.append((params, auc))
        for key in param_static:
            del params[key]

        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
        print("best:", sorted_results[0])

if __name__ == "__main__":
    main()
