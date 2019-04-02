"""Catboost 5-folds."""

import pandas as pd
import numpy as np
import torch
import random
import warnings

from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.modules.loss import BCEWithLogitsLoss

warnings.filterwarnings("ignore")


def feature_generator(df, vcs_train_test):
    for i in tqdm(range(200)):
        col = "var_" + str(i)

        vtraintest = vcs_train_test[col]
        t = vtraintest[df[col]].fillna(0).values

        df[col + '_train_test_sum_vcs'] = t
        df[col + '_train_test_sum_vcs_product'] = df[col] * t


def batch_iter(X, y, batch_size=64):
    """
    X: feature tensor (shape: num_instances x num_features)
    y: target tensor (shape: num_instances)
    """
    idxs = torch.randperm(X.size(0))
    if X.is_cuda:
        idxs = idxs.cuda()
    for batch_idxs in idxs.split(batch_size):
        yield X[batch_idxs], y[batch_idxs]


class NN(torch.nn.Module):

    def __init__(self, D_in=3, enc_out=30, features=200, random_seed=42):

        super(NN, self).__init__()
        self.layer = []
        layer_size = D_in

        for i in range(features):
            layer = torch.nn.Sequential(torch.nn.Linear(layer_size, enc_out // 2),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(enc_out // 2, enc_out),
                                        torch.nn.ReLU())
            setattr(self, 'layer_' + str(i), layer)

        self.linear3 = torch.nn.Linear(features * enc_out, 1)

    def forward(self, y):
        res = []
        for i in range(200):
            layer = getattr(self, 'layer_' + str(i))
            res.append(layer(y[:, i, :]))
        y = torch.cat(res, 1)
        y = self.linear3(y)
        return y


def main():
    random_seed = 42
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    input_path = "../input/"
    output_path = "../output/"

    print("torch:", torch.__version__)
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

    for col in train.columns:
        vcs_train_test[col] = train_test.loc[:, col].value_counts()

    feature_generator(train, vcs_train_test)
    feature_generator(test, vcs_train_test)

    scaler = StandardScaler()
    cols = train.columns

    X_train = pd.DataFrame(scaler.fit_transform(train), columns=cols)
    test = pd.DataFrame(scaler.transform(test), columns=cols)

    loss_f = BCEWithLogitsLoss()
    batch_size = 16384
    device = torch.device('cuda:3')

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(train))
    predictions = np.zeros(len(test))

    for fold_, (trn_idx, val_idx) in enumerate(skf.split(X_train.values, label.values)):
        print("Fold {}".format(fold_))

        train_tensors = []
        val_tensors = []
        test_tensors = []

        for fff in range(200):
            train_t = X_train.loc[trn_idx, [f'var_{fff}',
                                            f'var_{fff}_train_test_sum_vcs',
                                            f'var_{fff}_train_test_sum_vcs_product']].values
            val_t = X_train.loc[val_idx, [f'var_{fff}',
                                          f'var_{fff}_train_test_sum_vcs',
                                          f'var_{fff}_train_test_sum_vcs_product']].values
            test_t = test[[f'var_{fff}',
                           f'var_{fff}_train_test_sum_vcs',
                           f'var_{fff}_train_test_sum_vcs_product']].values
            train_tensors.append(torch.tensor(train_t, requires_grad=False,
                                              device=device, dtype=torch.float32))
            val_tensors.append(torch.tensor(val_t, requires_grad=False,
                                            device=device, dtype=torch.float32))

            test_tensors.append(torch.tensor(test_t, requires_grad=False,
                                             device=device, dtype=torch.float32))

        train_tensors = torch.cat(train_tensors, 1).view((-1, 200, 3))
        val_tensors = torch.cat(val_tensors, 1).view((-1, 200, 3))
        test_tensors = torch.cat(test_tensors, 1).view((-1, 200, 3))

        y_train_t = torch.tensor(label[trn_idx].values, requires_grad=False,
                                 device=device, dtype=torch.float32)
        y_val_t = torch.tensor(label[val_idx].values, requires_grad=False,
                               device=device, dtype=torch.float32)

        nn = NN().to(device)
        optimizer = Adam(params=nn.parameters(), lr=0.005)
        scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=5, min_lr=0.0001, verbose=True)
        best_AUC = 0
        early_stop = 0

        for epoch in tqdm(range(200)):
            dl = batch_iter(train_tensors, y_train_t, batch_size=batch_size)
            for data, label_t in dl:
                pred = nn(data)
                loss = loss_f(pred, torch.unsqueeze(label_t, -1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                val_pred = nn(val_tensors)
                AUC = roc_auc_score(label[val_idx].values, val_pred.detach().cpu().numpy())
                print('EPOCH {}'.format(epoch))
                print('LOSS: ', loss_f(val_pred, torch.unsqueeze(y_val_t, -1)).detach().cpu().numpy())
                print('AUC: ', AUC)
                scheduler.step(AUC)

                if AUC > best_AUC:
                    early_stop = 0
                    best_AUC = AUC
                    torch.save(nn, 'best_auc_nn.pkl')
                else:
                    early_stop += 1
                    print('SCORE IS NOT THE BEST. Early stop counter: {}'.format(early_stop))

                if early_stop == 15:
                    print('EARLY_STOPPING')
                    break
                print('=' * 50)

            best_model = torch.load('best_auc_nn.pkl')

        with torch.no_grad():
            oof[val_idx] = best_model(val_tensors).data.cpu().numpy().flatten()

            blobs = []
            for batch in torch.split(test_tensors, 20000):
                blob = best_model(batch).data.cpu().numpy().flatten()
                blobs.append(blob)
        predictions_test = np.concatenate(blobs)

        predictions += predictions_test / skf.n_splits

    auc = round(roc_auc_score(label, oof), 5)
    print("CV score: {:<8.5f}".format(auc))

    np.save(output_path + f"nn_{auc}_oof.npy", oof)
    np.save(output_path + f"nn_{auc}_test.npy", predictions)


if __name__ == "__main__":
    main()
