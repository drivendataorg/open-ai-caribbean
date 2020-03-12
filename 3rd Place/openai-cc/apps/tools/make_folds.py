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
        nfolds = config['make-folds']['nfolds']
        only_verified = config['make-folds']['only_verified']
        verified_in_train = config['make-folds']['verified_in_train']
        holdout_size = config['make-folds'].get('holdout')

    return Args()


def extend_from_json(train, geojson):
    geodata = {}
    for area_train in geojson:
        area_name = area_train['image_path']
        for features in area_train['features']:
            geodata[features['id']] = {
                'id': features['id'],
                'area_name': area_name,
                'coordinates': features['coordinates']
            }

    train = train.merge(json_normalize(list(train.id.map(geodata))), how='left', on='id')

    return train


def make_folds(path, nfolds, holdout_size, verified_in_train):
    HOLDOUT_FOLD_NAME = 777

    BAD_IDS = ['7a379232', '7a40099e', '7a29e812', '7a35a2d8', '7a36aade', '7a35862c', '7a29e330', '7a1e31e8',
               '7a2a721e', '7a305846', '7a277410', '7a2e4cea', '7a371e60', '7a39f572', '7a331bc6', '7a28603c',
               '7a344bd6', '7a41db5c']

    print(f'Making {nfolds} folds...')
    train = pd.read_csv(os.path.join(path, 'train_labels.csv'))
    train = train[~train.id.isin(BAD_IDS)]

    with open(os.path.join(path, 'train_labels.json')) as fp:
        train_json = json.load(fp)

    train = extend_from_json(train, train_json)

    train_cp = train.copy()
    train = train[train.verified]

    labels = np.argmax(train.iloc[:, 2:-2].values, axis=1)

    kfold = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=13)
    folds = np.zeros(len(train))
    for i, (_, val_ids) in enumerate(kfold.split(train, labels)):
        folds[val_ids] = i
    train['fold'] = folds

    if holdout_size is not None:
        holdout_ids = np.random.choice(train.index, int(holdout_size * len(train)))
        train.loc[holdout_ids, 'fold'] = HOLDOUT_FOLD_NAME

    if verified_in_train:
        extra = train_cp[~train_cp.verified]
        extra['fold'] = -1
        train = pd.concat([train, extra], axis=0)

    return train


def main():
    args = parse_args()
    df = make_folds(path=args.path,
                    nfolds=args.nfolds,
                    holdout_size=args.holdout_size,
                    verified_in_train=args.verified_in_train)
    save_path = os.path.join(args.path, 'folds.csv')
    df.to_csv(save_path, index=None)
    print(f'Succesfully saved to {save_path}')


if __name__ == '__main__':
    main()
