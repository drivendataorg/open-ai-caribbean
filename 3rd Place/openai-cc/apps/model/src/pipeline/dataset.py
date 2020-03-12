import os
import sys
import torch
import pandas as pd
import numpy as np
import json

sys.path.append(os.path.join(os.path.abspath('./'), './apps/model/src/pipeline/'))

from torch.utils.data import Dataset, DataLoader
from youtrain.factory import DataFactory
from transforms import *
from apps.visualize.src.utils import load_tiff, mask_map


class BaseDataset(Dataset):
    def __init__(self, image_dir, ids, transform):
        self.image_dir = image_dir
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        raise NotImplementedError


class TrainDataset(BaseDataset):
    def __init__(self, image_dir, ids, transform):
        super().__init__(image_dir, ids, transform)

        self.target_columns = [
            'concrete_cement',
            'healthy_metal',
            'incomplete',
            'irregular_metal',
            'other'
        ]

    def get_label(self, row):
        ids = np.argmax(row[self.target_columns].values)
        return np.array([ids])

    def _load_images(self):
        names = self.ids.area_name.unique()
        images = {}
        for name in names:
            images[name] = load_tiff(os.path.join(self.image_dir, name))

        return images

    def crop(self, image_name, coordinates):
        item = {'coordinates': coordinates}
        mask_crop = mask_map(item=item, tiff_path=os.path.join(self.image_dir, image_name))
        return mask_crop[:3].transpose([1, 2, 0])

    def __getitem__(self, index):
        coords = json.loads(self.ids.coordinates.iloc[index])
        image_name = self.ids.area_name.iloc[index]
        image = self.crop(image_name=image_name, coordinates=coords)
        cl = self.get_label(self.ids.iloc[index])
        result = self.transform(image=image)
        result['mask'] = torch.Tensor(cl)

        return result


class TestDataset(BaseDataset):
    def __init__(self, image_dir, ids, transform):
        super().__init__(image_dir, ids, transform)

        self.target_columns = [
            'concrete_cement',
            'healthy_metal',
            'incomplete',
            'irregular_metal',
            'other'
        ]

    def crop(self, image_name, coordinates):
        item = {'coordinates': coordinates}
        mask_crop = mask_map(item=item, tiff_path=os.path.join(self.image_dir, image_name))
        return mask_crop[:3].transpose([1, 2, 0])

    def __getitem__(self, index):
        coords = json.loads(self.ids.coordinates.iloc[index])
        image_name = self.ids.area_name.iloc[index]
        image = self.crop(image_name=image_name, coordinates=coords)
        result = self.transform(image=image)

        return result['image']


class TaskDataFactory(DataFactory):
    def __init__(self, params, paths, **kwargs):
        super().__init__(params, paths, **kwargs)
        self.fold = kwargs['fold']
        self._folds = None

    @property
    def data_path(self):
        return self.paths['path']

    def make_transform(self, stage, is_train=False):
        if is_train:
            transform = eval(stage['augmentation'])(**self.params['augmentation_params'])
        else:
            transform = test_transform(**self.params['augmentation_params'])
        return transform

    def build_sampler(self, folds):
        print(f'Sampler enable; samples in data: {len(folds)}')
        from collections import Counter
        def get_probas():
            vals = np.argmax(folds.iloc[:, 2:7].values, axis=1)
            vals2proba = np.argmax(folds[folds.fold != -1].iloc[:, 2:7].values, axis=1)
            counter = Counter(vals2proba)
            counter = {k: (v / np.sum(vals == k)) for k, v in counter.items()}
            return [counter[k] for k in vals]

        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            get_probas(),
            num_samples=len(folds[folds.fold != -1])-1,
            replacement=False)
        return sampler

    def make_dataset(self, stage, is_train):
        transform = self.make_transform(stage, is_train)
        ids = self.train_ids if is_train else self.val_ids
        return TrainDataset(
            image_dir=self.data_path,
            ids=ids,
            transform=transform)

    def make_loader(self, stage, is_train=False):
        dataset = self.make_dataset(stage, is_train)
        return DataLoader(
            dataset=dataset,
            batch_size=self.params['batch_size'],
            shuffle=is_train,
            drop_last=is_train,
            num_workers=self.params['num_workers'],
            pin_memory=torch.cuda.is_available(),
            # sampler=self.build_sampler(dataset.ids) if is_train else None
        )

    @property
    def folds(self):
        if self._folds is None:
            self._folds = pd.read_csv(os.path.join(self.data_path, 'folds.csv'))
        return self._folds

    @property
    def train_ids(self):
        return self.folds.loc[(self.folds['fold'] != self.fold) & (self.folds['fold'] != 777)]

    @property
    def val_ids(self):
        return self.folds.loc[self.folds['fold'] == self.fold]
