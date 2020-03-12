import os
import json
import numpy as np
import pandas as pd

from tqdm import tqdm
from .src.utils import transform_coordinates, found_epsg
from apps.utils import get_config


def parse_args():
    class Args:
        config = get_config(path=os.environ["OPENAI_CONFIG"])

        path = config['path']
        save = config['prepare-target']['save']

    return Args()


def calculate_target(path, meta, save=False):
    train = []
    test = []

    for _, row in tqdm(meta.iterrows(), total=len(meta)):
        epsg = found_epsg(os.path.join(path, row.image))

        features = transform_coordinates(
            path_to_geojson=os.path.join(path, row.train),
            epsg=epsg)

        train.append({
            'image_path': row.image,
            'features': features
        })

        if row.test is np.nan:
            continue

        features = transform_coordinates(
            path_to_geojson=os.path.join(path, row.test),
            epsg=epsg)

        test.append({
            'image_path': row.image,
            'features': features
        })

    if save:
        with open(os.path.join(path, 'train_labels.json'), 'w') as fp:
            json.dump(train, fp)

        with open(os.path.join(path, 'test_labels.json'), 'w') as fp:
            json.dump(test, fp)

    return train, test


def main():
    args = parse_args()
    meta = pd.read_csv(os.path.join(args.path, 'metadata.csv'))
    calculate_target(
        path=args.path,
        meta=meta,
        save=args.save
    )


if __name__ == '__main__':
    main()
