"""fast.ai NN 5-folds."""

import os
import sys

args = sys.argv
fold = int(args[1])
device = str(fold % 3)

os.environ["CUDA_VISIBLE_DEVICES"] = device

import fastai
import fastprogress

from fastai.basic_data import load_data
from fastai.tabular import *
from fastprogress import force_console_behavior
from os.path import isfile

#fastprogress.fastprogress.NO_BAR = True
master_bar, progress_bar = force_console_behavior()
fastai.basic_train.master_bar = master_bar
fastai.basic_train.progress_bar = progress_bar

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

        df[col + '_vcs'] = t
        df[col + '_vcs_prod'] = df[col] * t
        df[col + '_vcs_unique'] = np.array(t == 1).astype(int)

def auc_score(y_pred,y_true,tens=True):
    score = roc_auc_score(y_true, torch.sigmoid(y_pred)[:, 1])
    if tens:
        score = tensor(score)
    else:
        score = score
    return score

random_seed = 42

np.random.seed(random_seed)
random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)

print("fastai:", fastai.__version__)
print("torch:", torch.__version__)

path = Path('../output/')
bs = 2048

fname = path/f'data_{fold}_cont.pkl'
if not isfile(fname):
	print("Preparing data for fold", fold)

	input_path = "../input/"
	output_path = "../output/"

	train_df = pd.read_csv(input_path + 'train.csv.zip')
	train = train_df.drop(['ID_code'], axis=1)

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

	dep_var = "target"
	cat_names = []
	for i in range(200):
		cat_names += [#f"var_{i}",
		               f"var_{i}_vcs",
		               #f"var_{i}_vcs_prod",
		               f"var_{i}_vcs_unique"]
	cont_names = []
	for i in range(200):
		cont_names += [f"var_{i}",
		               #f"var_{i}_vcs",
		               f"var_{i}_vcs_prod",
		               #f"var_{i}_vcs_unique"
		               ]

	procs = [FillMissing, Categorify, Normalize]
	#print(len(df.columns),len(cont_names),len(cat_names))

	skf = StratifiedKFold(n_splits=5, random_state=2019, shuffle=True)

	i = 0
	for train_index, valid_index in skf.split(train, train[dep_var]):
	    if i == fold:
	    	break
	    i += 1

	data = (TabularList.from_df(train, path=path, cat_names=cat_names,
								cont_names=cont_names, procs=procs)
	                           .split_by_idx(valid_index)
	                           .label_from_df(cols=dep_var)
	                           .databunch(bs=bs))
	pickle.dump(data, open(fname, 'wb'), protocol=4)
else:
    print("Loading data of fold", fold)
    data = pickle.load(open(fname, 'rb'))

print("Building learner...")
learn = tabular_learner(data, layers=[20, 10],
                        ps=0.5,
                        emb_drop=0.5,
                        y_range=[0, 1],
                        use_bn=True,
                        metrics=[auc_score])
lr = 1e-2
for j in range(7):
	# if not isfile(path / f"models/model_{j}_{fold}_cont.pth"):
	print("Training model", j, fold)
	learn.fit_one_cycle(j + 1, lr / (1 + j), moms=(0.8, 0.7))
	learn.save(f'model_{j}_{fold}_cont')
	# else:
	# 	print("Loading model", j, fold)
	# 	learn.load(f'model_{j}_{fold}_cont')

if not isfile(f"../results/valid_{fold}_cont.csv.gz"):
	print("Predicting valid...")
	preds = learn.get_preds(ds_type=DatasetType.Valid)
	prob = [float(preds[0][i][1]) for i in range(len(preds[0]))]
	
	print("Saving predict...")
	df = pd.DataFrame({"prob": prob})
	df.to_csv(f"../results/valid_{fold}_cont.csv.gz", compression='gzip', index=False)

# if not isfile(f"../results/test_{fold}_cont.csv.gz"):
# 	print("Adding test...")
# 	dt = pd.read_hdf(path/'X_test.h5')
# 	dt = float_converter(dt)
# 	dt["label"] = -1
# 	_, cat_names, cont_names = col_names(dt)
# 	test = TabularList.from_df(dt, path=path, cat_names=cat_names, cont_names=cont_names)
# 	data.add_test(test)

# 	print("Predicting test...")
# 	preds = learn.get_preds(ds_type=DatasetType.Test)
# 	prob = [float(preds[0][i][1]) for i in range(len(preds[0]))]
	
# 	print("Saving predict...")
# 	df = pd.DataFrame({"prob":prob})
# 	df.to_csv(f"../results/test_{fold}_cont.csv.gz", compression='gzip', index=False)