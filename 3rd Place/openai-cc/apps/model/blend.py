import pandas as pd
import numpy as np

import os
import sys

sys.path.append(os.path.join(os.path.abspath('./'), './apps/model/src/pipeline/'))

from apps.utils import get_config
import warnings
warnings.filterwarnings('ignore')

from scipy.optimize import minimize
from sklearn.metrics import log_loss


def parse_args():
    class Args:
        config = get_config(path=os.environ["OPENAI_CONFIG"])

        path = config['path']
        pred_name = config['predictions-name']
        names = config['blend']['predictions']
        # blend_type = config['blend'].get('blend-type')
        # mode = config['blend']['mode']
        weights = config['blend'].get('weights')
        clipping = config['blend'].get('clipping')

    return Args()


target_columns = [
    'concrete_cement',
    'healthy_metal',
    'incomplete',
    'irregular_metal',
    'other'
]


def optimize(preds, train):
    func_to_average = lambda w: np.average(preds, weights=w, axis=0)
    func_to_optimize = lambda w: log_loss(train, func_to_average(w))
    return minimize(func_to_optimize, np.ones(len(preds)))


def blend(predictions, weights, train):
    train = train.loc[:, target_columns].values
    values = [pred.loc[:, target_columns].values for pred in predictions]
    predictions = predictions[0]
    if weights is None:
        weights = optimize(values, train).x
        print(weights)
        # exit()

    predictions.loc[:, target_columns] = np.average(values, weights=weights, axis=0)
    return predictions


def main():
    args = parse_args()
    predictions = [pd.read_csv(os.path.join(args.path, path)) for path in args.names]
    train = pd.read_csv(os.path.join(args.path, 'folds.csv'))
    if args.weights is None:
        train = predictions[0][['id']].merge(train, on='id')

    predictions = blend(predictions, args.weights, train)
    if args.clipping is not None:
        predictions.loc[:, target_columns] = predictions.loc[:, target_columns].clip(args.clipping[0], args.clipping[1])

    predictions.to_csv(os.path.join(args.path, args.pred_name))


if __name__ == '__main__':
    main()
