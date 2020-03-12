# coding: utf-8
__author__ = 'ZFTurbo: https://www.drivendata.org/users/ZFTurbo/'


from a00_common_functions import *
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold


def read_input_data():
    train = pd.read_csv(FEATURES_PATH + 'train_neigbours.csv')
    test = pd.read_csv(FEATURES_PATH + 'test_neigbours.csv')

    train['target'] = -1
    for i, c in enumerate(CLASSES):
        train.loc[train['roof_material'] == c, 'target'] = i

    # Add additional neighbours data
    d1 = pd.read_csv(FEATURES_PATH + 'neighbours_clss_distribution_10_train.csv')
    train = pd.merge(train, d1, on='id', how='left')
    d1 = pd.read_csv(FEATURES_PATH + 'neighbours_clss_distribution_10_test.csv')
    test = pd.merge(test, d1, on='id', how='left')

    # Add additional neighbours data
    d1 = pd.read_csv(FEATURES_PATH + 'neighbours_clss_distribution_100_train.csv')
    train = pd.merge(train, d1, on='id', how='left')
    d1 = pd.read_csv(FEATURES_PATH + 'neighbours_clss_distribution_100_test.csv')
    test = pd.merge(test, d1, on='id', how='left')

    # Add additional neighbours data
    d1 = pd.read_csv(FEATURES_PATH + 'neighbours_feat_10_train.csv')
    train = pd.merge(train, d1, on='id', how='left')
    d1 = pd.read_csv(FEATURES_PATH + 'neighbours_feat_10_test.csv')
    test = pd.merge(test, d1, on='id', how='left')

    # Add additional neighbours data
    d1 = pd.read_csv(FEATURES_PATH + 'neighbours_clss_distribution_radius_1000_train.csv')
    train = pd.merge(train, d1, on='id', how='left')
    d1 = pd.read_csv(FEATURES_PATH + 'neighbours_clss_distribution_radius_1000_test.csv')
    test = pd.merge(test, d1, on='id', how='left')

    # Add additional neighbours data
    d1 = pd.read_csv(FEATURES_PATH + 'neighbours_clss_distribution_radius_10000_train.csv')
    train = pd.merge(train, d1, on='id', how='left')
    d1 = pd.read_csv(FEATURES_PATH + 'neighbours_clss_distribution_radius_10000_test.csv')
    test = pd.merge(test, d1, on='id', how='left')

    # Add DenseNet121 data
    model_num = 1
    d1 = pd.read_csv(FEATURES_PATH + 'd121_kfold_valid_TTA_32_5.csv')
    d1.drop('real_answ', axis=1, inplace=True)
    for c in list(CLASSES):
        d1.rename(columns={c: c + '_model_{}'.format(model_num)}, inplace=True)
    train = pd.merge(train, d1, on='id', how='left')
    d1 = pd.read_csv(FEATURES_PATH + 'd121_v2_test_TTA_32_5.csv')
    for c in list(CLASSES):
        d1.rename(columns={c: c + '_model_{}'.format(model_num)}, inplace=True)
    test = pd.merge(test, d1, on='id', how='left')
    model_num += 1

    # Add IRV2 data
    d1 = pd.read_csv(FEATURES_PATH + 'irv2_kfold_valid_TTA_32_5.csv')
    d1.drop('real_answ', axis=1, inplace=True)
    for c in list(CLASSES):
        d1.rename(columns={c: c + '_model_{}'.format(model_num)}, inplace=True)
    train = pd.merge(train, d1, on='id', how='left')
    d1 = pd.read_csv(FEATURES_PATH + 'irv2_kfold_test_TTA_32_5.csv')
    for c in list(CLASSES):
        d1.rename(columns={c: c + '_model_{}'.format(model_num)}, inplace=True)
    test = pd.merge(test, d1, on='id', how='left')
    model_num += 1

    # Add IRV2 data (v2)
    d1 = pd.read_csv(FEATURES_PATH + 'irv2_kfold_v2_valid_TTA_32_5.csv')
    d1.drop('real_answ', axis=1, inplace=True)
    for c in list(CLASSES):
        d1.rename(columns={c: c + '_model_{}'.format(model_num)}, inplace=True)
    train = pd.merge(train, d1, on='id', how='left')
    d1 = pd.read_csv(FEATURES_PATH + 'irv2_kfold_v2_test_TTA_32_5.csv')
    for c in list(CLASSES):
        d1.rename(columns={c: c + '_model_{}'.format(model_num)}, inplace=True)
    test = pd.merge(test, d1, on='id', how='left')
    model_num += 1

    # Add EFFB4 data
    d1 = pd.read_csv(FEATURES_PATH + 'effnetb4_kfold_valid_TTA_32_5.csv')
    d1.drop('real_answ', axis=1, inplace=True)
    for c in list(CLASSES):
        d1.rename(columns={c: c + '_model_{}'.format(model_num)}, inplace=True)
    train = pd.merge(train, d1, on='id', how='left')
    d1 = pd.read_csv(FEATURES_PATH + 'effnetb4_kfold_test_TTA_32_5.csv')
    for c in list(CLASSES):
        d1.rename(columns={c: c + '_model_{}'.format(model_num)}, inplace=True)
    test = pd.merge(test, d1, on='id', how='left')
    model_num += 1

    # Add DenseNet169 data
    d1 = pd.read_csv(FEATURES_PATH + 'densenet169_kfold_valid_TTA_32_6.csv')
    d1.drop('real_answ', axis=1, inplace=True)
    for c in list(CLASSES):
        d1.rename(columns={c: c + '_model_{}'.format(model_num)}, inplace=True)
    train = pd.merge(train, d1, on='id', how='left')
    d1 = pd.read_csv(FEATURES_PATH + 'densenet169_kfold_test_TTA_32_6.csv')
    for c in list(CLASSES):
        d1.rename(columns={c: c + '_model_{}'.format(model_num)}, inplace=True)
    test = pd.merge(test, d1, on='id', how='left')
    model_num += 1

    # Add ResNet34 data
    d1 = pd.read_csv(FEATURES_PATH + 'resnet34_kfold_valid_TTA_32_5.csv')
    d1.drop('real_answ', axis=1, inplace=True)
    for c in list(CLASSES):
        d1.rename(columns={c: c + '_model_{}'.format(model_num)}, inplace=True)
    train = pd.merge(train, d1, on='id', how='left')
    d1 = pd.read_csv(FEATURES_PATH + 'resnet34_kfold_test_TTA_32_5.csv')
    for c in list(CLASSES):
        d1.rename(columns={c: c + '_model_{}'.format(model_num)}, inplace=True)
    test = pd.merge(test, d1, on='id', how='left')
    model_num += 1

    # Add SeResNext50 data
    d1 = pd.read_csv(FEATURES_PATH + 'seresnext50_kfold_valid_TTA_32_5.csv')
    d1.drop('real_answ', axis=1, inplace=True)
    for c in list(CLASSES):
        d1.rename(columns={c: c + '_model_{}'.format(model_num)}, inplace=True)
    train = pd.merge(train, d1, on='id', how='left')
    d1 = pd.read_csv(FEATURES_PATH + 'seresnext50_kfold_test_TTA_32_5.csv')
    for c in list(CLASSES):
        d1.rename(columns={c: c + '_model_{}'.format(model_num)}, inplace=True)
    test = pd.merge(test, d1, on='id', how='left')
    model_num += 1

    # Add ResNet50 data
    d1 = pd.read_csv(FEATURES_PATH + 'resnet50_kfold_valid_TTA_32_10.csv')
    d1.drop('real_answ', axis=1, inplace=True)
    for c in list(CLASSES):
        d1.rename(columns={c: c + '_model_{}'.format(model_num)}, inplace=True)
    train = pd.merge(train, d1, on='id', how='left')
    d1 = pd.read_csv(FEATURES_PATH + 'resnet50_kfold_test_TTA_32_10.csv')
    for c in list(CLASSES):
        d1.rename(columns={c: c + '_model_{}'.format(model_num)}, inplace=True)
    test = pd.merge(test, d1, on='id', how='left')
    model_num += 1

    features = list(test.columns.values)
    features.remove('id')

    return train, test, features


def get_params():
    params = {
        'target': 'target',
        'id' : 'id',
        'metric': 'log_loss',
    }
    params['metric_function'] = log_loss
    return params
