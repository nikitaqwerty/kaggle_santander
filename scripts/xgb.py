"""Catboost 5-folds."""

import pandas as pd
import numpy as np
import xgboost as xgb
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

    print("xgb:", xgb.__version__)
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
    early_stop_rounds = 100

    params = {'eval_metric': 'auc',
              'booster': 'gbtree',
              'tree_method': 'hist',
              'objective': 'binary:logistic',
              # 'subsample': 0.9,
              # 'colsample_bytree': 0.9,
              'eta': 0.03,
              'max_depth': 4,
              'seed': 42,
              'verbosity': 0}

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(train))
    predictions = np.zeros(len(test))
    d_test = xgb.DMatrix(test)

    for train_index, valid_index in skf.split(train, label):

        X_train = train.iloc[train_index]
        X_valid = train.iloc[valid_index]

        y_train = label.iloc[train_index]
        y_valid = label.iloc[valid_index]

        d_train = xgb.DMatrix(X_train, y_train)
        d_valid = xgb.DMatrix(X_valid, y_valid)

        model = xgb.train(params,
                          d_train,
                          rounds,
                          [(d_train, 'train'), (d_valid, 'eval')],
                          early_stopping_rounds=early_stop_rounds,
                          verbose_eval=50)

        best = model.best_iteration + 1
        oof[valid_index] = model.predict(d_valid, ntree_limit=best)
        predictions += model.predict(d_test, ntree_limit=best) / skf.n_splits

    auc = round(roc_auc_score(label, oof), 5)
    print("CV score: {:<8.5f}".format(auc))

    np.save(output_path + f"lgb_{auc}_oof.npy", oof)
    np.save(output_path + f"lgb_{auc}_test.npy", predictions)

if __name__ == "__main__":
    main()
