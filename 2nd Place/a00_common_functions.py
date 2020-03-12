# coding: utf-8
__author__ = 'ZFTurbo: https://www.drivendata.org/users/ZFTurbo/'

import numpy as np
import gzip
import pickle
import os
import glob
import time
import cv2
import datetime
import pandas as pd
from sklearn.metrics import fbeta_score
from sklearn.model_selection import KFold, train_test_split
from collections import Counter, defaultdict
from sklearn.metrics import accuracy_score, roc_auc_score
import random
import shutil
import operator
import pyvips
from PIL import Image
import platform
import json
import base64
import typing as t
import zlib


random.seed(2016)
np.random.seed(2016)

ROOT_PATH = os.path.dirname(os.path.realpath(__file__)) + '/'
INPUT_PATH = ROOT_PATH + 'input/'
OUTPUT_PATH = ROOT_PATH + 'modified_data/'
if not os.path.isdir(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)
MODELS_PATH = ROOT_PATH + 'models/'
if not os.path.isdir(MODELS_PATH):
    os.mkdir(MODELS_PATH)
CACHE_PATH = ROOT_PATH + 'cache/'
if not os.path.isdir(CACHE_PATH):
    os.mkdir(CACHE_PATH)
FEATURES_PATH = ROOT_PATH + 'features/'
if not os.path.isdir(FEATURES_PATH):
    os.mkdir(FEATURES_PATH)
if not os.path.isdir(FEATURES_PATH + 'oof/'):
    os.mkdir(FEATURES_PATH + 'oof/')
HISTORY_FOLDER_PATH = MODELS_PATH + "history/"
if not os.path.isdir(HISTORY_FOLDER_PATH):
    os.mkdir(HISTORY_FOLDER_PATH)
SUBM_PATH = ROOT_PATH + 'subm/'
if not os.path.isdir(SUBM_PATH):
    os.mkdir(SUBM_PATH)
ONLY_INFERENCE = True

CLASSES = ['concrete_cement', 'healthy_metal', 'incomplete', 'irregular_metal', 'other']
NUM_CLASSES = len(CLASSES)

def save_in_file(arr, file_name):
    pickle.dump(arr, gzip.open(file_name, 'wb+', compresslevel=3))


def load_from_file(file_name):
    return pickle.load(gzip.open(file_name, 'rb'))


def save_in_file_fast(arr, file_name):
    pickle.dump(arr, open(file_name, 'wb'))


def load_from_file_fast(file_name):
    return pickle.load(open(file_name, 'rb'))


def show_image(im, name='image'):
    cv2.imshow(name, im.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_image_rgb(im, name='image'):
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    cv2.imshow(name, im.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_resized_image(P, w=1000, h=1000):
    res = cv2.resize(P.astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    show_image(res)


def get_date_string():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")


def sort_dict_by_values(a, reverse=True):
    sorted_x = sorted(a.items(), key=operator.itemgetter(1), reverse=reverse)
    return sorted_x


def value_counts_for_list(lst):
    a = dict(Counter(lst))
    a = sort_dict_by_values(a, True)
    return a


def save_history_figure(history, path, columns=('acc', 'val_acc')):
    import matplotlib.pyplot as plt
    s = pd.DataFrame(history.history)
    plt.plot(s[list(columns)])
    plt.savefig(path)
    plt.close()


def read_single_image(path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return img


def read_image_bgr_fast(path):
    img2 = read_single_image(path)
    img2 = img2[:, :, ::-1]
    return img2


def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    number_size = 4.0
    if K.floatx() == 'float16':
         number_size = 2.0
    if K.floatx() == 'float64':
         number_size = 8.0

    total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes


def get_image_size(path):
    img = pyvips.Image.new_from_file(path, access='sequential')
    return img.height, img.width, img.bands


def preread_train_images():
    cache_path = OUTPUT_PATH + 'train_images.pkl'
    if not os.path.isfile(cache_path):
        files = glob.glob(OUTPUT_PATH + 'train_img/*.png')
        res = dict()
        for f in files:
            if '_mask.png' in f:
                continue
            id = os.path.basename(f)
            print('Go for: {}'.format(id))
            img1 = read_single_image(f)
            mask = cv2.imread(f[:-4] + '_mask.png', 0)
            mask = np.expand_dims(mask, axis=2)
            print(img1.shape, mask.shape)
            res[id] = np.concatenate((img1, mask), axis=2)
        save_in_file_fast(res, cache_path)
    else:
        res = load_from_file_fast(cache_path)
    return res


def preread_test_images():
    cache_path = OUTPUT_PATH + 'test_images.pkl'
    if not os.path.isfile(cache_path):
        files = glob.glob(OUTPUT_PATH + 'test_img/*.png')
        res = dict()
        for f in files:
            if '_mask.png' in f:
                continue
            id = os.path.basename(f)
            print('Go for: {}'.format(id))
            img1 = read_single_image(f)
            mask = cv2.imread(f[:-4] + '_mask.png', 0)
            mask = np.expand_dims(mask, axis=2)
            print(img1.shape, mask.shape)
            res[id] = np.concatenate((img1, mask), axis=2)
        save_in_file_fast(res, cache_path)
    else:
        res = load_from_file_fast(cache_path)
    return res


def get_kfold_split(folds, seed=None):
    from sklearn.model_selection import KFold
    if seed is None:
        cache_path = OUTPUT_PATH + 'kfold_{}.pkl'.format(folds)
    else:
        cache_path = OUTPUT_PATH + 'kfold_{}_seed_{}.pkl'.format(folds, seed)
    """
    For only inference mode we need to read PKL file. Otherwise 2nd level model won't work if KFold split changed for some reason)!
    """
    if not os.path.isfile(cache_path) and not ONLY_INFERENCE:
        train_ids = []
        train_answ = []
        valid_ids = []
        valid_answ = []
        for i in range(folds):
            train_ids.append([])
            train_answ.append([])
            valid_ids.append([])
            valid_answ.append([])

        train = pd.read_csv(OUTPUT_PATH + 'train.csv')
        unique_maps = train['train_path'].unique()
        print(unique_maps)
        for um in unique_maps:
            part = train[train['train_path'] == um]
            for c in CLASSES:
                part1 = part[part['roof_material'] == c]
                print('Part {} {}: {}'.format(um, c, len(part1)))
                if len(part1) == 0:
                    continue
                ids = part1['id'].values
                answ = part1['roof_material'].values
                if len(part1) < folds:
                    for total in range(folds):
                        train_ids[total].append(ids)
                        train_answ[total].append(answ)
                else:
                    kf = KFold(n_splits=folds, random_state=seed, shuffle=True)
                    total = 0
                    for train_f, valid_f in kf.split(ids):
                        train_ids[total].append(ids[train_f])
                        train_answ[total].append(answ[train_f])
                        valid_ids[total].append(ids[valid_f])
                        valid_answ[total].append(answ[valid_f])
                        print(len(train_f), len(valid_f))
                        total += 1

        for i in range(folds):
            train_ids[i] = np.concatenate(train_ids[i], axis=0)
            train_answ[i] = np.concatenate(train_answ[i], axis=0)

            valid_ids[i] = np.concatenate(valid_ids[i], axis=0)
            valid_answ[i] = np.concatenate(valid_answ[i], axis=0)

            print('Train files fold {}: {}'.format(i, len(train_ids[i])))
            print('Valid files fold {}: {}'.format(i, len(valid_ids[i])))
            print(train_ids[i].shape, valid_ids[i].shape)
            print('Intersection: {}'.format(len(set(train_ids[i]) & set(valid_ids[i]))))

        save_in_file_fast((train_ids, train_answ, valid_ids, valid_answ), cache_path)
    else:
        train_ids, train_answ, valid_ids, valid_answ = load_from_file_fast(cache_path)
    return train_ids, train_answ, valid_ids, valid_answ


def copy_best_model(models_path_init, fold_num, clean=True):
    models_path_best = models_path_init + 'best/'
    if not os.path.isdir(models_path_best):
        os.mkdir(models_path_best)
    files = glob.glob(models_path_init + '*_fold_{}_*.h5'.format(fold_num))
    best_model = ''
    best_loss = 1000
    for f in files:
        arr = os.path.basename(f).split('_')
        loss = float(arr[-5])
        if loss < best_loss:
            best_loss = loss
            best_model = f
    shutil.copy(best_model, models_path_best)
    if clean is True:
        for f in files:
            os.remove(f)


def get_best_models_for_cnn(models_dir, kfold_num):
    paths = []
    for kf in range(kfold_num):
        f = glob.glob(models_dir + 'best/*_fold_{}_*.h5'.format(kf))[0]
        paths.append(f)
    return paths
