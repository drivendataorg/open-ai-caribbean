import os
import pandas as pd
import json

from pandas.io.json import json_normalize

from apps.utils import get_config


def parse_args():
    class Args:
        config = get_config(path=os.environ["OPENAI_CONFIG"])

        path = config['path']
        prediction_name = config['predictions-name']

    return Args()


def add_missed(path, pred_name):
    target_columns = [
        'concrete_cement',
        'healthy_metal',
        'incomplete',
        'irregular_metal',
        'other'
    ]
    BAD_IDS = ['7a451c90',
         '7a4efc74',
         '7a4d32b8',
         '7a4ec4ac',
         '7a46856c',
         '7a46f6dc',
         '7a4ae9f4',
         '7a4cb770',
         '7a4715fe',
         '7a4b8850']
    test = pd.read_csv(os.path.join(path, pred_name))
    mean_vals = dict(test.loc[:, target_columns].mean())
    add_test = test.iloc[:len(BAD_IDS)].copy()
    add_test.id = BAD_IDS

    for col, val in mean_vals.items():
        add_test[col] = val
    test = pd.concat([test, add_test], axis=0)
    return test


def main():
    args = parse_args()
    df = add_missed(args.path, args.prediction_name)
    save_path = os.path.join(args.path, args.prediction_name + '_fixed.csv')

    # Re-order submit acc to submission_format file
    order = pd.read_csv(os.path.join(args.path, 'submission_format.csv'))
    needed_columns = order.columns
    df = order[['id']].merge(df, on='id', how='left')[needed_columns]
    assert len(df) == len(order)

    df.to_csv(save_path, index=None)
    print(f'Succesfully saved to {save_path}')


if __name__ == '__main__':
    main()
