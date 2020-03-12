import os
import pandas as pd
from sklearn.metrics import log_loss


def calculate_score(y_true, y_pred):
    return log_loss(y_true, y_pred)


def prepare_target_and_preds(path, preds_path, verified_only):
    target_columns = [
        'concrete_cement',
        'healthy_metal',
        'incomplete',
        'irregular_metal',
        'other'
    ]
    preds = pd.read_csv(preds_path)
    target = pd.read_csv(os.path.join(path, 'train_labels.csv'))
    if verified_only:
        print('\nVerified only enabled')
        target = target[target.verified]
        preds = preds[preds.id.isin(target.id)]
    target = target[target.id.isin(preds.id)]

    target = target.sort_values('id')
    preds = preds.sort_values('id')

    target = target.loc[:, target_columns].values
    preds = preds.loc[:, target_columns].values

    return target, preds, target_columns
