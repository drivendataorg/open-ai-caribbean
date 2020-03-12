import os
import json
import yaml


yaml.warnings({'YAMLLoadWarning': False})


def load_target(path):
    if not os.path.exists(os.path.join(path, 'train_labels.json')):
        raise Exception(f'Not found labels in {path}')

    with open(os.path.join(path, 'train_labels.json')) as fp:
        train = json.load(fp)

    return train


def get_config(path):
    with open(path, 'r') as stream:
        config = yaml.load(stream)
    return config
