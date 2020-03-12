import os
import pandas as pd

from importlib.machinery import SourceFileLoader


def load_model_module(model_module_name, model_module_path):
    print('-' * 40)
    print('READING MODEL-MODULE: {}'.format(model_module_name))
    model_module_file = os.path.join(model_module_path, '{}.py'.format(model_module_name))
    mm = SourceFileLoader('', model_module_file).load_module()
    return mm


def load_kfolds(dset_df, kfolds_file_path):
    # Read K-Folds split & merge to dset_df
    print('-' * 40)
    print('READING K-FOLDS FILE: {}'.format(kfolds_file_path))

    kfolds_df = pd.read_csv(kfolds_file_path)
    dset_df = pd.merge(dset_df, kfolds_df, how='left', left_on='id', right_on='id')

    return dset_df
