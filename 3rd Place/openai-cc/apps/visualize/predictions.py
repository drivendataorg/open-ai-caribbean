import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from .src.plot import draw_plot
from .src.utils import load_data

from apps.utils import get_config


def parse_args():
    class Args:
        config = get_config(path=os.environ["OPENAI_CONFIG"])

        path = config['path']
        predictions_name = config['predictions-name']
        ids = config['visualize']['ids']

    return Args()


def process_ids(path, preds_name, ids):
    def _get_label(preds, id):
        if preds is None:
            return None

        columns = list(preds.columns)[1:]

        row = preds[preds.id == id][columns].values
        return columns[np.argmax(row)]

    data = load_data(path)

    preds = pd.read_csv(os.path.join(path, preds_name)) if preds_name else None
    for id in tqdm(ids):
        pred_label = _get_label(preds, id) if preds else None
        draw_plot(
            path=path,
            id=id,
            data=data,
            pred_label=pred_label
        )


def main():
    args = parse_args()
    process_ids(
        path=args.path,
        preds_name=args.predictions_name,
        ids=args.ids
    )


if __name__ == '__main__':
    main()
