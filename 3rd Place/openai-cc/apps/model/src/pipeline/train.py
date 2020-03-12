from pathlib import Path

import cv2
import torch

from .dataset import TaskDataFactory
from .utils import create_callbacks

from .youtrain.factory import Factory
from .youtrain.runner import Runner

import warnings
warnings.filterwarnings('ignore')

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def run_training(config, paths, fold, device):
    config['train_params']['name'] = f'{config["train_params"]["name"]}/{fold}'
    config['train_params']['name_save'] = paths["dumps"]['name_save']
    config['train_params']['save_dir'] = Path(paths['dumps']['path']) / config['train_params']['name']

    factory = Factory(config['train_params'])

    data_factory = TaskDataFactory(
        params=config['data_params'],
        paths=paths,
        fold=fold
    )

    callbacks = create_callbacks(
        name=config['train_params']['name'],
        dumps=paths['dumps'],
        name_save=paths["dumps"]["name_save"],
        monitor_metric=config['train_params']['metrics'][-1]
    )

    trainer = Runner(
        stages=config['stages'],
        factory=factory,
        callbacks=callbacks,
        device=device,
    )

    report = trainer.fit(data_factory)
    return report
