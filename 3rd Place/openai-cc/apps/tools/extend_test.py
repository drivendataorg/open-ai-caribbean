import os
import pandas as pd
import json

from pandas.io.json import json_normalize

from apps.utils import get_config


def parse_args():
    class Args:
        config = get_config(path=os.environ["OPENAI_CONFIG"])

        path = config['path']

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


def extend_test(path):
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
    test = pd.read_csv(os.path.join(path, 'submission_format.csv'))
    test = test[~test.id.isin(BAD_IDS)]
    with open(os.path.join(path, 'test_labels.json')) as fp:
        train_json = json.load(fp)

    test = extend_from_json(test, train_json)
    return test


def main():
    args = parse_args()
    df = extend_test(args.path)
    save_path = os.path.join(args.path, 'test_ids.csv')
    df.to_csv(save_path, index=None)
    print(f'Succesfully saved to {save_path}')


if __name__ == '__main__':
    main()
