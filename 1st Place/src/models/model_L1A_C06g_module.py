import numpy as np
import pandas as pd

import pretrainedmodels
import pretrainedmodels.utils as ptm_utils
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.geodata_utils import get_data_dict, get_dataset_df
from src.utils.misc import AttrDict
from src.data.roof_dataset import RoofDataset
from src.utils.image_transformations import trim
from src.data.data_feeders import DataFeed

MODEL_TYPE = 'L1A'
MODEL_VERSION = 'C'
TRAINING_VERSION = '06g'

args = AttrDict({})

# Data
args.update({
    'use_only_verified': True,  # Whether to use only verified data to train the model
    'add_location': False,

    'train_split_ratio': 0.091,  # Ratio split train/validation (if needed)
    'random_split_seed': 999,  # Random seed for splitting training set

    'correct_train_samples': None,
    'thershold_for_corrections': None,  # None or float
    'validate_with_corrected_labels': False,  # Use original labels or corrected to validate
})
# Model
args.update({
    'basemodel_name': 'dpn92',
    'pretrained': 'imagenet+5k',  # 'imagenet' or 'imagenet+background' or 'imagenet+5k'
    'freeze_basemodel': False,  # Indicate first layer to train, True for freeze all, False none
    'outputs_as_onehotencoder': False,  # If output is class id or a vector with the class encoded
    'mixup': True,  # Apply mixup https://arxiv.org/pdf/1710.09412.pdf
    'mixup_alpha': 2.0,
    'mixup_method': 'default|inbatch',  # default, losses, inbatch
})
# Transformations
SIZE = 224
args.update({
    'transformations': {
        'train': transforms.Compose([
            transforms.RandomRotation((0, 360), expand=True),
            transforms.Lambda(lambda x: trim(x)),
            transforms.Lambda(lambda x: transforms.CenterCrop((max(x.size), max(x.size)))(x)),
            transforms.Resize(SIZE),
            transforms.RandomAffine(0, scale=(0.6, 1.1)),
            transforms.RandomCrop(SIZE, pad_if_needed=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1),
            # transforms.ToTensor(),
            # transforms.RandomErasing(p=0.5, scale=(1/8, 1/4), ratio=(0.5, 2.0), value=0, inplace=False),
            # transforms.ToPILImage(),
        ]),
        'valid': transforms.Compose([
            transforms.Lambda(lambda x: trim(x)),
            transforms.Lambda(lambda x: transforms.CenterCrop((max(x.size), max(x.size)))(x)),
            transforms.Resize(SIZE),
        ]),
        'fold': transforms.Compose([
            transforms.Lambda(lambda x: trim(x)),
            transforms.Lambda(lambda x: transforms.CenterCrop((max(x.size), max(x.size)))(x)),
            transforms.Resize(SIZE),
        ]),
        'test': transforms.Compose([
            transforms.Lambda(lambda x: trim(x)),
            transforms.Lambda(lambda x: transforms.CenterCrop((max(x.size), max(x.size)))(x)),
            transforms.Resize(SIZE),
        ]),
    }
})

# Detect if we have a GPU available
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
PARALLEL_TRAINING = False
NB_GPU = max(1, torch.cuda.device_count()) if PARALLEL_TRAINING else 1

# Execution
TRAIN_BATCH_SIZE = 48 * NB_GPU
VIRTUAL_BATCH_SIZE = None  # Accumulate gradients, Int or None. It may not work properly
args.update({
    'max_train_epochs': 40,
    'train_batch_size': TRAIN_BATCH_SIZE,
    'valid_batch_size': TRAIN_BATCH_SIZE,
    'test_batch_size': TRAIN_BATCH_SIZE * 1,
    'virtual_batch_size': VIRTUAL_BATCH_SIZE,
    'num_workers': 8 * NB_GPU,
    'parallel_training': PARALLEL_TRAINING,
})

# Model
args.update({
    'ouput_name': f"{MODEL_TYPE}_{MODEL_VERSION}{TRAINING_VERSION}"
})

ARGS = args

BASEMODEL_FEATURES = {
    'inceptionv4': (1536, lambda model: nn.Sequential(*list(model.children())[:-2], )),
    'dpn92': (2688, lambda model: nn.Sequential(*list(model.children())[:-1], )),
    'se_resnext101_32x4d': (2048, lambda model: nn.Sequential(*list(model.children())[:-2], )),
}


def get_dset(path_to_data, args=ARGS):
    print('-' * 40)
    print("READING DATASET")

    # Get dataset
    dset_df = get_dataset_df(path_to_data, verbose=False)
    annot_dict = {}

    # Filter dataset
    dset_df['unverified'] = False
    dset_df['valid'] = False
    if args.use_only_verified:
        print("* Using only verified data")
        dset_df.loc[dset_df.verified == False, 'unverified'] = True
        dset_df.loc[dset_df.verified == False, 'train'] = False
    print(f"Dataset shape: {dset_df.shape})")
    print(
        f"Training: {dset_df.train.sum()} | Validation: {dset_df.valid.sum()} | Testing: {dset_df.test.sum()} | "
        f"Unverified: {dset_df.unverified.sum()}")

    # Labels
    nb_classes = dset_df.class_id.max() + 1
    print(f'Number of classes (based on class_id): {nb_classes}')
    print(f'Number of unique classes in dataset: {len(dset_df[~dset_df.test].class_id.unique())}')
    annot_dict['nb_classes'] = nb_classes

    # Dictionary with class names
    classes = dset_df[dset_df.train].class_id.unique()
    names = []
    for iclass in classes:
        name = dset_df[dset_df.class_id == iclass].roof_material.unique()
        assert len(name) == 1
        names.append(name[0])
    classId_to_name = {s1: s2 for s1, s2 in zip(classes, names)}
    name_to_classId = {s2: s1 for s1, s2 in classId_to_name.items()}
    annot_dict['classId_to_name'] = classId_to_name

    # load predicted labels for train data
    dset_df = dset_df.assign(original_class_id=dset_df.class_id)
    if args.correct_train_samples is not None:
        print(f"Using predicted labels from file: {args.correct_train_samples}")
        val_df = pd.read_csv(args.correct_train_samples, index_col=0)
        val_df = pd.DataFrame({'roof_material': [val_df.columns[s1] for s1 in val_df.values.argmax(axis=1)],
                               'prob': val_df.values.max(axis=1)}, index=val_df.index)

        trn_ids = dset_df.id[dset_df.train]
        trn_lab = dset_df.roof_material[dset_df.train]
        trn_lab = trn_lab[trn_ids.isin(val_df.index)]
        trn_ids = trn_ids[trn_ids.isin(val_df.index)]

        val_df = val_df.loc[trn_ids]
        val_df = val_df.assign(true_lab=trn_lab.values)
        val_df = val_df[val_df.roof_material != val_df.true_lab]
        if args.thershold_for_corrections is not None:
            filter_val_ids = val_df.index[val_df.prob < args.thershold_for_corrections]
            val_df = val_df[~val_df.index.isin(filter_val_ids)]
            print(f"Filtering predictions with probability >= {args.thershold_for_corrections}: {len(val_df)}")

        val_ids = dset_df.id[dset_df.id.isin(val_df.index)].values
        val_labels = val_df.loc[val_ids, 'roof_material']
        val_classes = val_labels.map(lambda x: name_to_classId[x])

        dset_df.loc[dset_df.id.isin(val_df.index), 'roof_material'] = val_labels.values
        dset_df.loc[dset_df.id.isin(val_df.index), 'class_id'] = val_classes.values

        print(
            f"Training: {dset_df.train.sum()} | Validation: {dset_df.valid.sum()} | Testing: {dset_df.test.sum()} | "
            f"Unverified {dset_df.unverified.sum()}")

    return dset_df, annot_dict


def _generate_data_feeders(df_dict, load_function, transformations, preprocess, phases, args_dict):
    return {s1: DataFeed(df_dict[s1], load_function[s1],
                         x_transform_func=transformations[s1],
                         x_preprocess_func=preprocess,
                         y_preprocess_func=None,
                         **args_dict[s1]) for s1 in phases}


def _generate_data_loaders(date_feeder, phases, args_dict):
    return {s1: DataLoader(date_feeder[s1], **args_dict[s1]) for s1 in phases}


def _worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def get_dataloaders(path_to_data, datasets, args=ARGS, nb_classes=5):
    data_dict = get_data_dict(path_to_data)
    train_stages = datasets.keys()

    ## Data reader classes
    roof_data_reader = {
        'train': RoofDataset(datasets['train'], data_dict, buffer=1, mask_image=True, plot_polygons=False),
        'fold': RoofDataset(datasets['fold'], data_dict, buffer=1, mask_image=True, plot_polygons=False),
        'test': RoofDataset(datasets['test'], data_dict, buffer=1, mask_image=True, plot_polygons=False),
    }
    roof_data_reader = {
        'train': RoofDataset(datasets['train'], data_dict, buffer=1, mask_image=True, plot_polygons=False,
                             add_location=args.add_location,
                             return_ohe=args.outputs_as_onehotencoder, nb_classes=nb_classes),
        'fold': RoofDataset(datasets['fold'], data_dict, buffer=1, mask_image=True, plot_polygons=False,
                            add_location=args.add_location,
                            return_ohe=args.outputs_as_onehotencoder, nb_classes=nb_classes),
        'test': RoofDataset(datasets['test'], data_dict, buffer=1, mask_image=True, plot_polygons=False,
                            add_location=args.add_location,
                            return_ohe=args.outputs_as_onehotencoder, nb_classes=nb_classes),
    }

    if args.validate_with_corrected_labels and args.correct_train_samples is not None:
        print("Validating with corrected pseudo-labels")
        roof_data_reader['valid'] = RoofDataset(datasets['valid'], data_dict, buffer=1, mask_image=True,
                                                plot_polygons=False, add_location=args.add_location,
                                                return_ohe=args.outputs_as_onehotencoder, nb_classes=nb_classes)
    else:
        print("Validating with original labels")
        roof_data_reader['valid'] = RoofDataset(datasets['valid'], data_dict, buffer=1, mask_image=True,
                                                annot_column='original_class_id',
                                                plot_polygons=False, add_location=args.add_location,
                                                return_ohe=args.outputs_as_onehotencoder, nb_classes=nb_classes)

    ## Load images function
    load_function = {
        'train': lambda x: roof_data_reader['train'].read(x),
        'valid': lambda x: roof_data_reader['valid'].read(x),
        'fold': lambda x: roof_data_reader['fold'].read(x),
        'test': lambda x: roof_data_reader['test'].read(x),
    }

    ## Data Feeders
    img_pre_process_tt = transforms.Compose([
        transforms.ToTensor(),
        ptm_utils.ToSpaceBGR(
            pretrainedmodels.pretrained_settings[args.basemodel_name][args.pretrained]['input_space'] == 'BGR'),
        ptm_utils.ToRange255(
            max(pretrainedmodels.pretrained_settings[args.basemodel_name][args.pretrained]['input_range']) == 255),
        transforms.Normalize(mean=pretrainedmodels.pretrained_settings[args.basemodel_name][args.pretrained]['mean'],
                             std=pretrainedmodels.pretrained_settings[args.basemodel_name][args.pretrained]['std'])])

    def preprocess(PIL_imgs):
        imgs = [img_pre_process_tt(s) for s in PIL_imgs]
        return torch.stack(imgs).to(torch.float32)

    transformations = args.transformations
    data_feeder_args = {
        'train': {},
        'valid': {},
        'fold': {'predict': True},
        'test': {'predict': True},
    }
    # Data Feeders
    data_feeders = _generate_data_feeders(datasets, load_function, transformations, preprocess, train_stages,
                                          data_feeder_args)

    ## Data Loader
    data_loader_args = {
        "train": {'batch_size': args.train_batch_size, 'shuffle': True, 'worker_init_fn': _worker_init_fn, },
        "valid": {'batch_size': args.valid_batch_size, 'shuffle': False, 'worker_init_fn': _worker_init_fn, },
        "fold": {'batch_size': args.test_batch_size, 'shuffle': False, 'worker_init_fn': _worker_init_fn, },
        "test": {'batch_size': args.test_batch_size, 'shuffle': False, 'worker_init_fn': _worker_init_fn, },
    }
    if 'cuda' in DEVICE:
        for phase, dct in data_loader_args.items():
            dct.update({'pin_memory': True, 'num_workers': args.num_workers})
    # Data Loaders
    data_loaders = _generate_data_loaders(data_feeders, train_stages, data_loader_args)

    return data_loaders


class NNModel(nn.Module):
    def __init__(self, basemodel, num_ftrs, nb_classes, add_features=False):
        super(NNModel, self).__init__()

        self.features = basemodel
        if isinstance(num_ftrs, list):
            self.num_ftrs = num_ftrs[0]
            self.ext_ftrs = num_ftrs[1]
        else:
            self.num_ftrs = num_ftrs
            self.ext_ftrs = 0
        self.add_features = add_features
        self.pooling = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        ])
        if self.add_features:
            self.classifier = nn.Sequential(*[
                nn.Linear(self.num_ftrs + self.ext_ftrs, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(64, nb_classes),
            ])
        else:
            self.classifier = nn.Sequential(*[
                nn.Linear(self.num_ftrs, nb_classes),
            ])

        self.softmax = nn.Softmax(dim=-1)

        self.output_predictions = False

    def forward(self, x):
        if self.add_features:
            x, x1 = x[0], x[1]
        x = self.features(x)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        if self.add_features:
            x = torch.cat((x, x1), -1)
        x = self.classifier(x)

        if self.output_predictions:
            return self.softmax(x)
        else:
            return x


def _get_model(nb_classes, args=ARGS):
    print('-' * 80)
    print(f"LOADING MODEL")
    print(f"Using device: {DEVICE}")
    # Get basemodel
    num_classes = 1001 if args.pretrained == 'imagenet+background' else 1000
    basemodel = pretrainedmodels.__dict__[args.basemodel_name](num_classes=num_classes, pretrained=args.pretrained)
    basemodel = BASEMODEL_FEATURES[args.basemodel_name][1](basemodel)
    num_ftrs = BASEMODEL_FEATURES[args.basemodel_name][0]

    if args.freeze_basemodel is not None and args.freeze_basemodel is not False:
        if args.freeze_basemodel is True:
            print("Freezing base model")
            for param in basemodel.parameters():
                param.requires_grad = False
        else:
            print("Params to learn:")
            train = False
            for name, param in basemodel.named_parameters():
                if args.freeze_basemodel in name or train:
                    param.requires_grad = True
                    train = True
                    print("\t", name)
                else:
                    param.requires_grad = False

    model = NNModel(basemodel, num_ftrs, nb_classes)

    data_parallel = False
    if args.parallel_training:
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
            data_parallel = True

    return model


def get_learner(nb_classes, args=ARGS):
    model = _get_model(nb_classes, args=args)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    if args.outputs_as_onehotencoder:
        criterion = nn.BCEWithLogitsLoss()

        from src.metrics import pytorch_metrics
        metrics = [pytorch_metrics.CrossEntropyLoss(one_hot_encoding=True), ]
    else:
        criterion = nn.CrossEntropyLoss()

        from src.metrics import pytorch_metrics
        metrics = [pytorch_metrics.CategoricalAccuracy(), pytorch_metrics.TopKAccuracy(top_k=2)]

    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True, threshold=0.001,
                                  min_lr=1e-5)

    from src.pytorch.early_stoppers import EarlyStopping
    early_stopper = EarlyStopping(patience=6, verbose=True, delta=0.0001, save_model_path=None)

    from src.pytorch.wrappers import PyTorchNN_vA as PyTorchNN
    pytorchmodel = PyTorchNN(model, optimizer, criterion, metrics, scheduler, early_stopper, device=DEVICE,
                             virtual_batch_size=args.virtual_batch_size, mixup=args.mixup, mixup_alpha=args.mixup_alpha,
                             mixup_method=args.mixup_method)

    return pytorchmodel
