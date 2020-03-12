PATH_TO_ROOT = './'
import sys

sys.path.append(PATH_TO_ROOT)

# Base
import os
import pandas as pd
import numpy as np
import click

from sklearn.metrics import log_loss

# Execution
from src.data.geodata_utils import get_dataset_df

# User parameters
DATA_DIR = './data'  # Path from project root
PREDICTIONS_PATH = 'predictions'


@click.command()
@click.option('--model_id')
@click.option('--debug', is_flag=True)
def main(model_id, debug):
    MODEL_ID = 'L1A_' + model_id if not debug else 'debug_L1A_' + model_id
    print('-'*50)
    print(f'- Concatenating L1 predictions for {MODEL_ID}')
    print('-' * 50)

    # Dataset
    dset_df = get_dataset_df(os.path.join(PATH_TO_ROOT, DATA_DIR))
    dset_labels = dset_df[['id', 'roof_material']]
    dset_labels.set_index('id', inplace=True, drop=True)
    dset_classes = dset_df[['id', 'class_id']]
    dset_classes.set_index('id', inplace=True, drop=True)

    # Concatenate predictions
    files = [i for i in os.listdir(os.path.join(PATH_TO_ROOT, PREDICTIONS_PATH)) if i.startswith(MODEL_ID + "_fold_")]
    train_df = []
    metrics = []
    for file in files:
        tmp_df = pd.read_csv(os.path.join(PATH_TO_ROOT, PREDICTIONS_PATH, file))
        train_df.append(tmp_df.copy())
        # Evaluate
        tmp_df.set_index('id', inplace=True, drop=True)
        y_pred_a = [tmp_df.columns[s1] for s1 in tmp_df.values.argmax(axis=1)]
        y_true_a = dset_labels.loc[tmp_df.index, 'roof_material'].values
        y_pred_l = tmp_df.iloc[:, 0:5].values
        y_true_l = dset_classes.loc[tmp_df.index, 'class_id'].values.astype(int)
        metrics.append(
            (file.split('.')[0].split('fold_')[1], (y_pred_a == y_true_a).mean(), log_loss(y_true_l, y_pred_l)))
        print(f"File: {file} | Accuracy: {(y_pred_a == y_true_a).mean():.4} | "
              f"Log-loss: {log_loss(y_true_l, y_pred_l):.4}")

    train_df = pd.concat(train_df, axis=0)
    train_df.set_index('id', inplace=True, drop=True)
    print(train_df.shape)
    print(train_df.head(2))

    metrics = list(zip(*metrics))
    metrics = pd.DataFrame({'Accuracy': metrics[1], 'Log-Loss': metrics[2]}, index=metrics[0])
    metrics = metrics.sort_index().transpose()
    print(metrics)

    # Evaluate
    y_pred = [train_df.columns[s1] for s1 in train_df.values.argmax(axis=1)]
    y_true = dset_labels.loc[train_df.index, 'roof_material'].values
    print(f"Accuracy: {(y_pred == y_true).mean():.4}")
    y_pred = train_df.iloc[:, 0:5].values
    y_true = dset_classes.loc[train_df.index, 'class_id'].values.astype(int)
    print(f"Log-loss: {log_loss(y_true, y_pred):.4}")

    # Save predictions
    filepath = os.path.join(PATH_TO_ROOT, PREDICTIONS_PATH, MODEL_ID + f"_folds.csv.gz")
    print(f"Saving FOLDS predictions: {filepath}")
    train_df.to_csv(filepath, index=True)

    # Concatenate predictions
    filepath = os.path.join(PATH_TO_ROOT, DATA_DIR, f'submission_format.csv')
    sub_df = pd.read_csv(filepath, index_col=0)

    files = [i for i in os.listdir(os.path.join(PATH_TO_ROOT, PREDICTIONS_PATH)) if i.startswith(MODEL_ID + "_test")]
    test_df = []
    metrics = []
    for file in files:
        tmp_df = pd.read_csv(os.path.join(PATH_TO_ROOT, PREDICTIONS_PATH, file))
        test_df.append(tmp_df.copy())

    test_df = pd.concat(test_df, axis=0)
    print(test_df.shape)
    test_df = test_df.groupby('id').mean()
    print(test_df.shape, test_df.sum(axis=1).sum())
    test_df.loc[:, :] = np.divide(test_df.values, test_df.sum(axis=1).values[:, np.newaxis])
    print(test_df.shape, test_df.sum(axis=1).sum())
    test_df = test_df.loc[sub_df.index]
    print(test_df.head(2))

    # Save predictions
    filepath = os.path.join(PATH_TO_ROOT, PREDICTIONS_PATH, MODEL_ID + f"_test.csv.gz")
    print(f"Saving TEST predictions: {filepath}")
    test_df.to_csv(filepath, index=True)

    print('')


if __name__ == '__main__':
    main()
