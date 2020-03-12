import os
import numpy as np
import pandas as pd
import json

from sklearn.model_selection import StratifiedKFold
from pandas.io.json import json_normalize

from apps.utils import get_config


def parse_args():
    class Args:
        config = get_config(path=os.environ["OPENAI_CONFIG"])

        path = config['path']
        pred_name = config['predictions-name']
        verified_gap = 0.25

    return Args()


def main():
    target_columns = [
        'concrete_cement',
        'healthy_metal',
        'incomplete',
        'irregular_metal',
        'other'
    ]

    args = parse_args()

    folds = pd.read_csv(os.path.join(args.path, 'folds.csv'))
    folds_verified = pd.read_csv(os.path.join(args.path, 'folds_verified.csv'))
    folds_verified = folds_verified[~folds_verified.verified]

    preds_verified = pd.read_csv(os.path.join(args.path, args.pred_name))
    folds_verified = folds_verified.sort_values('id')
    preds_verified = preds_verified.sort_values('id')
    assert len(folds_verified) == len(preds_verified)

    max_mat = preds_verified.loc[:, target_columns].values * \
              folds_verified.loc[:, target_columns].values

    max_mat = max_mat.max(axis=1)
    mask = (max_mat > 1 - args.verified_gap)
    print(f"Added {mask.sum()} samples")
    samples2add = folds_verified[mask]
    folds = pd.concat([folds, samples2add], axis=0)
    folds.to_csv(os.path.join(args.path, 'folds.csv'), index=False)


if __name__ == '__main__':
    main()
