import pandas as pd
import numpy as np
import torch
import sys
import time
import random
import warnings

from tqdm import tqdm
from random import choice
from os.path import isfile
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from torch.optim import Adam, RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.modules.loss import BCEWithLogitsLoss

from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

warnings.filterwarnings("ignore")


class UpsamplingPreprocessor:
    def __init__(self, times=10, neg_class_balancer=2):
        self.times = times
        self.neg_class_balancer = neg_class_balancer
        self.random_seed = 42

    # Data augmentation
    def augment_class(self, X):
        X_new = X.copy()

        for c in range(X.shape[1]):
            np.random.shuffle(X_new[:, c])

        return X_new

    def augment(self, X, y):
        np.random.seed(self.random_seed)

        t_pos = self.times
        t_neg = self.times // self.neg_class_balancer

        X_pos_orig = X[y == 1]
        X_neg_orig = X[y == 0]
        X_pos = np.zeros((t_pos, *X_pos_orig.shape), dtype=X.dtype)
        X_neg = np.zeros((t_neg, *X_neg_orig.shape), dtype=X.dtype)

        for i in range(t_pos):
            X_pos[i] = self.augment_class(X_pos_orig)

        for i in range(t_neg):
            X_neg[i] = self.augment_class(X_neg_orig)

        X_pos = np.vstack(X_pos)
        X_neg = np.vstack(X_neg)
        y_pos = np.ones(X_pos.shape[0])
        y_neg = np.zeros(X_neg.shape[0])
        X = np.vstack((X, X_pos, X_neg))
        y = np.concatenate((y, y_pos, y_neg))

        return X, y

    def fit_transform(self, X, y=None):
        var_cols = ['var_{}'.format(x) for x in range(200)]
        X_augmented, y = self.augment(X.values, y)
        return pd.DataFrame(X_augmented, columns=var_cols), y

    def transform(self, X):
        return X


def generate_features(df, vcs_train_test, cols):
    for col in tqdm(cols):
        vtraintest = vcs_train_test[col]
        t = vtraintest[df[col]].fillna(0).values

        df[col + '_1_flag'] = (t == 1).astype(int)


class NN(torch.nn.Module):
    def __init__(self, D_in=5, features=200, enc_out=30, enc_hidden_layer_k=2, use_dropout=False, use_BN=False):
        super(NN, self).__init__()
        self.layer = []
        layer_size = D_in
        h_size = int(enc_out / enc_hidden_layer_k)
        for i in range(features):
            if use_dropout and not use_BN:
                layer = torch.nn.Sequential(torch.nn.Linear(layer_size, h_size),
                                            torch.nn.Linear(h_size, enc_out),
                                            torch.nn.ReLU(),
                                            torch.nn.Dropout()
                                            )
            elif not use_dropout and not use_BN:
                layer = torch.nn.Sequential(torch.nn.Linear(layer_size, h_size),
                                            torch.nn.Linear(h_size, enc_out),
                                            torch.nn.ReLU()
                                            )
            elif not use_dropout and use_BN:
                layer = torch.nn.Sequential(torch.nn.Linear(layer_size, h_size),
                                            torch.nn.Linear(h_size, enc_out),
                                            torch.nn.ReLU(),
                                            torch.nn.BatchNorm1d(num_features=enc_out)
                                            )
            elif use_dropout and use_BN:
                layer = torch.nn.Sequential(torch.nn.Linear(layer_size, h_size),
                                            torch.nn.Linear(h_size, enc_out),
                                            torch.nn.ReLU(),
                                            torch.nn.BatchNorm1d(num_features=enc_out),
                                            torch.nn.Dropout(),
                                            )
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
def main():

    def evaluate_nn(**new_params):
        new_params['batch_size'] = int(new_params['batch_size'])
        new_params['nn_encoder_out'] = int(new_params['nn_encoder_out'])
        new_params['lr_sheduler_patience'] = int(new_params['lr_sheduler_patience'])
        new_params['upsampling_times'] = int(new_params['upsampling_times'])
        new_params['upsampling_class_balancer'] = int(new_params['upsampling_class_balancer'])
        new_params['use_bn'] = new_params['use_bn'] > 0.5
        new_params['use_dropout'] = new_params['use_dropout'] > 0.5

        params.update(new_params)
        print('Training with params: {}'.format(new_params))

        ups = UpsamplingPreprocessor(params['upsampling_times'], params['upsampling_class_balancer'])

        loss_f = BCEWithLogitsLoss()

        batch_size = params['batch_size']
        N_IN = 2

        gpu = torch.device(f'cuda:{gpu_num % 4}')
        cpu = torch.device('cpu')

        folds = StratifiedKFold(n_splits=params['n_splits'], shuffle=True, random_state=42)
        oof = np.zeros(len(train))

        for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, label)):
            # print("Fold {}".format(fold_))

            X_train, Train_label = ups.fit_transform(train.loc[trn_idx], label.loc[trn_idx])
            X_val, Val_label = train.loc[val_idx], label.loc[val_idx]
            generate_features(X_train, vcs_train_test, cols)
            generate_features(X_val, vcs_train_test, cols)
            cols_new = X_train.columns
            scaler = StandardScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=cols_new)
            X_val = pd.DataFrame(scaler.transform(X_val), columns=cols_new)

            break

        train_tensors = []
        val_tensors = []

        for fff in range(200):
            cols_to_use = [f'var_{fff}', f'var_{fff}_1_flag']
            train_t = X_train.loc[:, cols_to_use].values
            val_t = X_val.loc[:, cols_to_use].values
            train_tensors.append(torch.tensor(train_t, requires_grad=False, device=cpu, dtype=torch.float32))
            val_tensors.append(torch.tensor(val_t, requires_grad=False, device=cpu, dtype=torch.float32))

        train_tensors = torch.cat(train_tensors, 1).view((-1, 200, N_IN))
        val_tensors = torch.cat(val_tensors, 1).view((-1, 200, N_IN))

        try:
            y_train_t = torch.tensor(Train_label, requires_grad=False, device=cpu, dtype=torch.float32)
        except:
            y_train_t = torch.tensor(Train_label.values, requires_grad=False, device=cpu, dtype=torch.float32)

        try:
            y_val_t = torch.tensor(Val_label, requires_grad=False, device=cpu, dtype=torch.float32)
        except:
            y_val_t = torch.tensor(Val_label.values, requires_grad=False, device=cpu, dtype=torch.float32)

        nn = NN(D_in=N_IN,
                enc_out=params['nn_encoder_out'],
                enc_hidden_layer_k=params['enc_hidden_layer_k'],
                use_dropout=params['use_dropout'],
                use_BN=params['use_bn']).to(gpu)

        if params['optimizer'] == 'adam':
            optimizer = Adam(params=nn.parameters(), lr=params['lr'])
        elif params['optimizer'] == 'RMSprop':
            optimizer = RMSprop(params=nn.parameters(), lr=params['lr'])

        scheduler = ReduceLROnPlateau(optimizer, 'max', factor=params['lr_sheduler_factor'],
                                      patience=params['lr_sheduler_patience'],
                                      min_lr=params['lr_sheduler_min_lr'], verbose=True)

        best_AUC = 0
        early_stop = 0

        for epoch in tqdm(range(params['max_epoch'])):
            nn.train()
            dl = batch_iter(train_tensors, y_train_t, batch_size=batch_size)
            for data, label_t in dl:
                pred = nn(data.to(gpu))
                loss = loss_f(pred, torch.unsqueeze(label_t.to(gpu), -1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                nn.eval()
                blobs = []
                for batch in torch.split(val_tensors, batch_size):
                    blob = nn(batch.to(gpu)).data.cpu().numpy().flatten()
                    blobs.append(blob)
                val_pred = np.concatenate(blobs)
                AUC = roc_auc_score(label[val_idx], val_pred)
                # print('EPOCH {}'.format(epoch))
                # print('LOSS: ', loss_f(torch.tensor(val_pred), y_val_t))
                # print('AUC: ', AUC)
                scheduler.step(AUC)

                if AUC > best_AUC:
                    early_stop = 0
                    best_AUC = AUC
                    torch.save(nn, output_path + f'best_auc_nn_{gpu_num}.pkl')
                else:
                    early_stop += 1
                    # print('SCORE IS NOT THE BEST. Early stop counter: {}'.format(early_stop))

                if early_stop == params['early_stop_wait']:
                    # print(f'EARLY_STOPPING NOW, BEST AUC = {best_AUC}')
                    break
                # print('=' * 50)

            best_model = torch.load(output_path + f'best_auc_nn_{gpu_num}.pkl')

        with torch.no_grad():
            best_model.eval()
            blobs = []
            for batch in torch.split(val_tensors, batch_size):
                blob = best_model(batch.to(gpu)).data.cpu().numpy().flatten()
                blobs.append(blob)

            oof[val_idx] = np.concatenate(blobs)

            auc = round(roc_auc_score(Val_label, oof[val_idx]), 5)
            return auc

    gpu_num = int(sys.argv[1])
    random_seed = (int(time.time()) * (gpu_num + 1)) % (2 ** 31 - 1)

    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    params = {'optimizer': 'adam',
              'n_splits': 10,
              'lr_sheduler_min_lr': 0.0001,
              'max_epoch': 9999,
              'early_stop_wait': 20}

    print("\nSEED:", random_seed)
    print("GPU:", gpu_num, "\n")

    input_path = "../input/"
    output_path = "../output/"

    print("torch:", torch.__version__)
    print("loading data...")

    train_df = pd.read_csv(input_path + 'train.csv.zip')

    label = train_df.target
    train = train_df.drop(['ID_code', 'target'], axis=1)
    cols = train.columns

    test = pd.read_csv(input_path + 'test.csv.zip')
    test = test.drop(['ID_code'], axis=1)

    test_filtered = pd.read_pickle(input_path + 'test_filtered.pkl')
    test_filtered = test_filtered.loc[:, train.columns]

    train_test = pd.concat([train, test_filtered]).reset_index(drop=True)

    vcs_train_test = {}

    for col in tqdm(train.columns):
        vcs_train_test[col] = train_test.loc[:, col].value_counts()

    generate_features(test, vcs_train_test, cols)

    bounds = {'batch_size': (4096, 16384),
              'nn_encoder_out': (10, 100),
              'enc_hidden_layer_k': (0.5, 4.0),
              'lr': (0.001, 0.01),
              'use_dropout': (0, 1),
              'use_bn': (0, 1),
              'lr_sheduler_factor': (0.1, 0.9),
              'lr_sheduler_patience': (3, 15),
              'upsampling_times': (3, 20),
              'upsampling_class_balancer': (2, 10)}

    bo = BayesianOptimization(evaluate_nn, pbounds=bounds, random_state=gpu_num)

    log_file = output_path + f"hpo_logs_{gpu_num}.json"
    if isfile(log_file):
        load_logs(bo, logs=[log_file])

    logger = JSONLogger(path=log_file)
    bo.subscribe(Events.OPTMIZATION_STEP, logger)

    bo.maximize(init_points=30, n_iter=1000)
    result = bo.res

if __name__ == "__main__":
    main()
