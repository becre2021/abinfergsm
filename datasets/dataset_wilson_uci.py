import pandas as pd
import numpy as np
import torch
from typing import Callable, Dict, Union
import time
import datetime
import traceback
import os
from scipy.io import loadmat
import json
import sys

sys.path.append('./../')




data_base_path = './datasets/'

def load_dataset(name: str):
    """Helper method to load a given UCI dataset
    Loads data from .mat files at path <data_base_path>/uci/<name>.mat
    """
    #mat = loadmat(os.path.join(data_base_path, 'uci', name, '{}.mat'.format(name)))
    mat = loadmat(os.path.join(data_base_path, 'uci_wilson', name, '{}.mat'.format(name)))
    
    [n, d] = mat['data'].shape
    df = pd.DataFrame(mat['data'],
                      columns=list(range(d-1))+['target'])
    df.columns = [str(c) for c in df.columns]
    df = df.reset_index()

    df['target'] = df['target'] - df['target'].mean()
    df['target'] = df['target']/(df['target'].std())

    df = df.dropna(axis=1, how='all')

    return df


def get_datasets():
    return get_small_datasets() + get_medium_datasets() + get_big_datasets()


def get_small_datasets():
    return ['challenger', 'fertility', 'concreteslump', 'autos', 'servo',
     'breastcancer', 'machine', 'yacht', 'autompg', 'housing', 'forest',
     'stock', 'pendulum', 'energy']


def get_medium_datasets():
    return ['concrete', 'solar', 'airfoil',
     'wine', 'gas', 'skillcraft', 'sml', 'parkinsons', 'pumadyn32nm']


def get_big_datasets():
    return ['pol', 'elevators', 'bike', 'kin40k', 'protein', 'tamielectric',
     'keggdirected', 'slice', 'keggundirected', '3droad', 'song',
     'buzz', 'houseelectric']


def format_timedelta(delta):
    d = delta.days
    h = delta.seconds // (60*60)
    m = (delta.seconds - h*(60*60)) // 60
    s = (delta.seconds - h*(60*60) - m*60)
    return '{}d {}h {}m {}s'.format(d, h, m, s)


def _determine_folds(split, dataset):
    """Determine the indices where folds begin and end."""
    n_per_fold = int(np.floor(len(dataset) * split))
    n_folds = int(round(1 / split))
    remaining = len(dataset) - n_per_fold * n_folds
    fold_starts = [0]
    for i in range(n_folds):
        if i < remaining:
            fold_starts.append(fold_starts[i] + n_per_fold + 1)
        else:
            fold_starts.append(fold_starts[i] + n_per_fold)
    return fold_starts


def _access_fold(dataset, fold_starts, fold):
    """Pull out the test and train set of a dataset using existing fold division"""
    train = dataset.iloc[0:fold_starts[fold]]  # if fold=0, none before fold
    test = dataset.iloc[fold_starts[fold]:fold_starts[fold + 1]]
    train = pd.concat([train, dataset.iloc[fold_starts[fold + 1]:]])
    return train, test


def _normalize_by_train(train, test):
    """Mean and std normalize using mean and std of the train set."""
    train = train.copy()
    test = test.copy()
    cols = list(train.columns)
    features = [x for x in cols if (x != 'target' and x.lower() != 'index')]
    mu = train[features + ['target']].mean()

    train.loc[:, features + ['target']] -= mu
    test.loc[:, features + ['target']] -= mu
    for f in features + ['target']:
        sigma = train[f].std()
        if sigma > 0:
            train.loc[:, f] /= sigma
            test.loc[:, f] /= sigma
    return train, test , features



def _load_ucidataset_wilson(dataset, split :float ,fold : int , exp2 : bool):
    dataset = load_dataset(dataset)
    fold_starts = _determine_folds(split, dataset)
    n_folds = len(fold_starts) - 1    
    train, test = _access_fold(dataset, fold_starts, fold)
    train, test ,features = _normalize_by_train(train, test)    
    
    ## train. test
    if exp2:
        trainX = torch.tensor(train[features].values, dtype=torch.double).contiguous()
        trainY = torch.tensor(train['target'].values, dtype=torch.double).contiguous().view(-1,1)
        testX = torch.tensor(test[features].values, dtype=torch.double).contiguous()
        testY = torch.tensor(test['target'].values, dtype=torch.double).contiguous().view(-1,1)       
        return trainX.cuda(),testX.cuda(),trainY.cuda(),testY.cuda() 
    else:
        # train. validation, test
        trainX = torch.tensor(train[features].values, dtype=torch.double).contiguous()
        trainY = torch.tensor(train['target'].values, dtype=torch.double).contiguous().view(-1,1)
        valX = torch.tensor(test[features].values, dtype=torch.double).contiguous()[::2]
        valY = torch.tensor(test['target'].values, dtype=torch.double).contiguous()[::2].view(-1,1)    
        testX = torch.tensor(test[features].values, dtype=torch.double).contiguous()[1::2]
        testY = torch.tensor(test['target'].values, dtype=torch.double).contiguous()[1::2].view(-1,1)   
        return trainX.cuda(),valX.cuda(),testX.cuda(),trainY.cuda(),valY.cuda(),testY.cuda()    #x_train, x_val, x_test, y_train, y_val,y_test

