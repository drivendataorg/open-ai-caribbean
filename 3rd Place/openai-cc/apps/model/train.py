import os
import torch

from apps.utils import get_config
from apps.model.src.pipeline.train import run_training


def parse_args():
    class Args:
        config = get_config(path=os.environ["OPENAI_CONFIG"])

        path = config['path']
        train_config = config['model']

    return Args()


def main():
    args = parse_args()

    paths = {
        'path': args.path,
        'dumps': {
            'path': os.path.join(args.path, 'dumps'),
            'weights': 'weights',
            'logs': 'logs',
            'name_save': 'model'
        },
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    name = args.train_config["train_params"]["name"]
    for fold in args.train_config['folds']:
        args.train_config["train_params"]["name"] = name # reset name to avoid adding /0/1/2/..
        print(f'Start training fold: {fold}')
        run_training(
            config=args.train_config,
            paths=paths,
            fold=fold,
            device=device
        )


if __name__ == '__main__':
    main()
