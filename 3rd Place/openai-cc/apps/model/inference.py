import numpy as np

import os
import sys
import glob

sys.path.append(os.path.join(os.path.abspath('./'), './apps/model/src/pipeline/'))

import cv2
import pydoc
import torch
import pandas as pd
from tqdm import tqdm

from dataset import TestDataset
from transforms import test_transform
from torch.utils.data import DataLoader
from youtrain.utils import get_config
from apps.utils import get_config
import warnings
warnings.filterwarnings('ignore')

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def parse_args():
    class Args:
        config = get_config(path=os.environ["OPENAI_CONFIG"])

        path = config['path']
        train_config = config['model']
        tta = config['inference']['tta']
        ids = config['inference']['ids']
        clips = config['inference'].get('clipping')

        folds = os.listdir(os.path.join(path, 'dumps', train_config['train_params']['name']))
        mode = config['inference']['mode']

        verified = config['inference'].get('verified')
        # weights = config['inference']['weights']

    return Args()


class PytorchInference:
    def __init__(self, device, activation='sigmoid'):
        self.device = device
        self.activation = activation

    @staticmethod
    def to_numpy(images):
        return images.data.cpu().numpy()

    def run_one_predict(self, model, images):
        predictions = model(images)
        if self.activation == 'sigmoid':
            predictions = predictions.sigmoid()
        elif self.activation == 'softmax':
            predictions = predictions.softmax(dim=1)
        return predictions

    @staticmethod
    def flip_tensor_lr(images):
        invert_indices = torch.arange(images.data.size()[-1] - 1, -1, -1).long()
        return images.index_select(3, invert_indices.cuda())

    @staticmethod
    def flip_tensor_tb(images):
        invert_indices = torch.arange(images.data.size()[-2] - 1, -1, -1).long()
        return images.index_select(2, invert_indices.cuda())

    def predict(self, model, loader, tta=False):
        model = model.to(self.device).eval()

        with torch.no_grad():
            for data in loader:
                images = data.to(self.device)
                predictions = self.run_one_predict(model, images)
                if tta:
                    predictions += self.run_one_predict(model, self.flip_tensor_lr(images))
                    predictions += self.run_one_predict(model, self.flip_tensor_tb(images))
                    predictions += self.run_one_predict(model, self.flip_tensor_lr(self.flip_tensor_tb(images)))
                    predictions /= 4

                for prediction in predictions:
                    prediction = np.moveaxis(self.to_numpy(prediction), 0, -1)
                    yield prediction


def main():
    target_columns = [
        'concrete_cement',
        'healthy_metal',
        'incomplete',
        'irregular_metal',
        'other'
    ]

    args = parse_args()
    config = args.train_config
    path = args.path
    model_name = config['train_params']['model']
    model = pydoc.locate(model_name)(**config['train_params']['model_params'])

    predictions = []
    for fold in args.folds:
        fold = int(fold)

        model_name = config['train_params']['name']
        path2weights = os.path.join(path, 'dumps', model_name, str(fold))
        weights = sorted(os.listdir(path2weights))[0]
        path2weights = os.path.join(path2weights, weights)
        print(f'Inference fold: {fold} from checkpoint: {path2weights}')

        model.load_state_dict(torch.load(path2weights)['state_dict'])

        ids = pd.read_csv(os.path.join(path, args.ids))

        if args.verified is not None and args.verified:
            ids = ids[~ids.verified]
        elif args.mode == 'train':
            ids = ids[ids.fold == fold]

        dataset = TestDataset(
                image_dir=path,
                ids=ids,
                transform=test_transform(**config['data_params']['augmentation_params'])
        )

        loader = DataLoader(
                dataset=dataset,
                batch_size=config['data_params']['batch_size'],
                shuffle=False,
                drop_last=False,
                num_workers=config['data_params']['num_workers'],
                pin_memory=torch.cuda.is_available()
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inferencer = PytorchInference(device, activation='softmax')

        preds = []
        for pred in tqdm(inferencer.predict(model, loader, args.tta), total=len(dataset)):
            preds.append(pred)
        preds = np.array(preds)

        if args.clips is not None:
            preds = np.clip(preds, a_min=args.clips[0], a_max=args.clips[1])

        ids.loc[:, target_columns] = preds
        predictions.append(ids)

    if args.mode == 'train':
        predictions = pd.concat(predictions, axis=0)
    elif args.mode == 'test':
        preds2average = np.mean([pred.loc[:, target_columns].values for pred in predictions], axis=0)
        predictions = predictions[0]
        predictions.loc[:, target_columns] = preds2average
    # preds2average = np.mean([pred.loc[:, target_columns].values for pred in predictions], axis=0)
    # predictions = predictions[0]
    # predictions.loc[:, target_columns] = preds2average

    if args.verified:
        save_name = f'predictions_verified_{model_name}.csv'
    else:
        save_name = f'predictions_{args.mode}_{model_name}.csv'

    predictions.to_csv(os.path.join(path, save_name), index=False)


if __name__== '__main__':
    main()
