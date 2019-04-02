"""Catboost 5-folds."""

import pandas as pd
import numpy as np
import catboost
import warnings

from tqdm import tqdm
from sklearn.preprocessing import scale
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier, Pool

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

    print("catboost:", catboost.__version__)
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

    params = {
        'task_type': 'GPU',
        'iterations': 20000,
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'random_seed': 4242,
        "learning_rate": 0.03,
        "l2_leaf_reg": 3.0,
        'bagging_temperature': 1,
        'random_strength': 1,
        'depth': 4,
        'border_count': 128}

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(train))
    predictions = np.zeros(len(test))

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, label.values)):
        print("Fold {}".format(fold_))
        trn_data = Pool(train.iloc[trn_idx], label=label.iloc[trn_idx])
        val_data = Pool(train.iloc[val_idx], label=label.iloc[val_idx])
        clf = CatBoostClassifier(**params)
        clf.fit(trn_data,
                eval_set=val_data,
                use_best_model=True,
                verbose=500,
                early_stopping_rounds=300)
        oof[val_idx] = clf.predict_proba(train.iloc[val_idx])[:, 1]
        predictions += clf.predict_proba(test)[:, 1] / folds.n_splits

    auc = round(roc_auc_score(label, oof), 5)
    print("CV score: {:<8.5f}".format(auc))

    np.save(output_path + f"cat_{auc}_oof.npy", oof)
    np.save(output_path + f"cat_{auc}_test.npy", predictions)

if __name__ == "__main__":
    main()
