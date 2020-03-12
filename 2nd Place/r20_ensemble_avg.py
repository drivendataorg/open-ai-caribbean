# coding: utf-8
__author__ = 'ZFTurbo: https://www.drivendata.org/users/ZFTurbo/'


from a00_common_functions import *


def ensemble(subms):
    total = None
    weight_sum = 0
    for subm_path_test, weight in subms:
        print('Read {} (weight {})'.format(subm_path_test, weight))
        s1 = pd.read_csv(subm_path_test)
        # s1.sort_values(by='id', inplace=True)
        # s1.reset_index(drop=True, inplace=True)
        weight_sum += weight
        if total is None:
            total = s1.copy()
            total[CLASSES] *= weight
        else:
            total[CLASSES] += weight * s1[CLASSES]

    total[CLASSES] /= weight_sum
    total.to_csv(SUBM_PATH + 'submission.csv'.format(len(subms)), index=False)


def ensemble_train(subms):
    total = None
    weight_sum = 0
    for subm_path_test, weight in subms:
        subm_path_train = subm_path_test[:-17] + '_train.csv'
        print('Read {} (weight {})'.format(subm_path_train, weight))
        s1 = pd.read_csv(subm_path_train)
        s1.sort_values(by='id', inplace=True)
        s1.reset_index(drop=True, inplace=True)
        weight_sum += weight
        if total is None:
            total = s1.copy()
            total[CLASSES] *= weight
        else:
            total[CLASSES] += weight * s1[CLASSES]

    total[CLASSES] /= weight_sum
    total.to_csv(SUBM_PATH + 'submission_train.csv'.format(len(subms)), index=False)


if __name__ == '__main__':
    subm_list = [
        (SUBM_PATH + 'catboost_ensemble.csv', 1),
        (SUBM_PATH + 'xgboost_ensemble.csv', 1),
        (SUBM_PATH + 'lightgbm_ensemble.csv', 1),
    ]
    # ensemble_train(subm_list)
    ensemble(subm_list)

'''
XGB (0.3700) + CatB (0.3700) = 0.3689
XGB (0.386864) + CatB (0.387795) + LGB (0.396322) = LB: 0.3641 
XGB (0.375957) + CatB (0.373275) + LGB (0.379433) = LB: 0.3556
XGB (0.375957) + CatB (0.373275) + LGB (0.379433) + Keras (0.399213) = LB: 0.3566
Pseudolabels: XGB (0.270457) + CatB (0.268235) + LGB (0.273755) LB: 0.3846
XGB (0.370199) + CatB (0.374865) + LGB (0.373893) = LB: 0.3618
XGB (0.374172) + CatB (0.378113) + LGB (0.378917) = LB: 0.3562
XGB (0.374635) + CatB (0.377802) + LGB (0.379240) = LB: 0.3540
XGB (0.373778) + CatB (0.378157) + LGB (0.378481) = LB: 0.3542
'''