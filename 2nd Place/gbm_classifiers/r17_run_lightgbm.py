# coding: utf-8
__author__ = 'ZFTurbo: https://www.drivendata.org/users/ZFTurbo/'


from a00_common_functions import *
from gbm_classifiers.a01_read_data import *


random.seed(2018)
np.random.seed(2018)


def print_importance(features, gbm, prnt=True):
    max_report = 100
    importance_arr = sorted(list(zip(features, gbm.feature_importance())), key=lambda x: x[1], reverse=True)
    s1 = 'Importance TOP {}: '.format(max_report)
    for d in importance_arr[:max_report]:
        s1 += str(d) + ', '
    if prnt:
        print(s1)
    return importance_arr


def get_kfold_split(folds_number, len_train, target, random_state):
    train_index = list(range(len_train))
    folds = StratifiedKFold(n_splits=folds_number, shuffle=True, random_state=random_state)
    ret = []
    for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train_index, target)):
        ret.append([trn_idx, val_idx])
    return ret


def create_lightgbm_model(train, features, params):
    import lightgbm as lgb
    import matplotlib.pyplot as plt
    target_name = params['target']

    print('LightGBM version: {}'.format(lgb.__version__))
    start_time = time.time()

    unique_target = np.array(sorted(train[target_name].unique()))
    print('Target length: {}: {}'.format(len(unique_target), unique_target))

    required_iterations = 13
    overall_train_predictions = np.zeros((len(train), len(unique_target)), dtype=np.float32)
    overall_importance = dict()

    model_list = []
    for iter1 in range(required_iterations):
        # Debug
        num_folds = random.randint(4, 6)
        random_state = 10
        learning_rate = random.choice([0.05, 0.06, 0.1])
        num_leaves = random.choice([16, 24, 32])
        feature_fraction = random.choice([0.8, 0.85, 0.9])
        bagging_fraction = random.choice([0.8, 0.85, 0.9])
        boosting_type = 'gbdt'
        # boosting_type = 'dart'
        min_data_in_leaf = 1
        # max_bin = 511
        bagging_freq = 5
        drop_rate = 0.05
        skip_drop = 0.5
        max_drop = 1

        if 1:
            params_lgb = {
                'task': 'train',
                'boosting_type': boosting_type,
                # 'objective': 'regression',
                # 'metric': {'rmse'},
                'objective': 'multiclass',
                'num_class': 5,
                'metric': {'multi_logloss'},
                'device': 'cpu',
                'gpu_device_id': 1,
                'num_leaves': num_leaves,
                'learning_rate': learning_rate,
                'feature_fraction': feature_fraction,
                'bagging_fraction': bagging_fraction,
                'min_data_in_leaf': min_data_in_leaf,
                'bagging_freq': bagging_freq,
                # 'max_bin': max_bin,
                'drop_rate': drop_rate,
                'boost_from_average': True,
                'skip_drop': skip_drop,
                'max_drop': max_drop,
                # 'lambda_l1': 5,
                # 'lambda_l2': 5,
                'feature_fraction_seed': random_state + iter1,
                'bagging_seed': random_state + iter1,
                'data_random_seed': random_state + iter1,
                'verbose': 0,
                'num_threads': 6,
            }
        log_str = 'LightGBM iter {}. PARAMS: {}'.format(iter1, sorted(params_lgb.items()))
        print(log_str)
        num_boost_round = 10000
        early_stopping_rounds = 100

        ret = get_kfold_split(num_folds, len(train), train[target_name].values, 821 + iter1)
        full_single_preds = np.zeros((len(train), len(unique_target)), dtype=np.float32)
        fold_num = 0
        for train_index, valid_index in ret:
            fold_num += 1
            print('Start fold {}'.format(fold_num))
            X_train = train.loc[train_index].copy()
            X_valid = train.loc[valid_index].copy()
            y_train = X_train[target_name]
            y_valid = X_valid[target_name]

            print('Train data:', X_train.shape)
            print('Valid data:', X_valid.shape)

            # Exclude unverified from validation
            if 1:
                bad_ids = train[train['verified'] == False]['id'].values
                # condition = ~np.isin(train_ids, bad_ids)
                # train_ids = train_ids[condition]
                # train_answ = train_answ[condition]
                condition = ~np.isin(X_valid['id'], bad_ids)
                X_valid_excluded = X_valid[condition]
                y_valid_excluded = y_valid[condition]
                print('Exclude bad IDs from validation...')
                print('Valid data:', X_valid_excluded.shape)

            lgb_train = lgb.Dataset(X_train[features].values, y_train)
            lgb_eval = lgb.Dataset(X_valid_excluded[features].values, y_valid_excluded, reference=lgb_train)

            gbm = lgb.train(params_lgb, lgb_train, num_boost_round=num_boost_round,
                            early_stopping_rounds=early_stopping_rounds, valid_sets=[lgb_eval], verbose_eval=50)

            imp = print_importance(features, gbm, True)
            model_list.append(gbm)

            for i in imp:
                if i[0] in overall_importance:
                    overall_importance[i[0]] += i[1] / num_folds
                else:
                    overall_importance[i[0]] = i[1] / num_folds

            pred = gbm.predict(X_valid[features].values, num_iteration=gbm.best_iteration)
            full_single_preds[valid_index] += pred

            pred1 = gbm.predict(X_valid_excluded[features].values, num_iteration=gbm.best_iteration)
            score = params['metric_function'](y_valid_excluded, pred1)
            print('Fold {} score: {:.6f}'.format(fold_num, score))

        # Exclude unverified from validation
        if 1:
            bad_ids = train[train['verified'] == False]['id'].values
            # condition = ~np.isin(train_ids, bad_ids)
            # train_ids = train_ids[condition]
            # train_answ = train_answ[condition]
            condition = ~np.isin(train['id'], bad_ids)
            full_single_preds_excluded = full_single_preds[condition]
            target_excluded = train[condition][target_name].values
            print('Exclude bad IDs from validation...')
            print('Valid data:', X_valid_excluded.shape)

        score = params['metric_function'](target_excluded, full_single_preds_excluded)
        overall_train_predictions += full_single_preds
        print('Score iter {}: {:.6f} Time: {:.2f} sec'.format(iter1, score, time.time() - start_time))

    overall_train_predictions /= required_iterations
    for el in overall_importance:
        overall_importance[el] /= required_iterations
    imp = sort_dict_by_values(overall_importance)
    names = []
    values = []
    print('Total importance count: {}'.format(len(imp)))
    output_features = 100
    for i in range(min(output_features, len(imp))):
        print('{}: {:.6f}'.format(imp[i][0], imp[i][1]))
        names.append(imp[i][0])
        values.append(imp[i][1])

    if 0:
        fig, ax = plt.subplots(figsize=(10, 25))
        ax.barh(list(range(min(output_features, len(imp)))), values, 0.4, color='green', align='center')
        ax.set_yticks(list(range(min(output_features, len(imp)))))
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        plt.subplots_adjust(left=0.47)
        plt.savefig('debug.png')

    # Exclude unverified from validation
    if 1:
        bad_ids = train[train['verified'] == False]['id'].values
        # condition = ~np.isin(train_ids, bad_ids)
        # train_ids = train_ids[condition]
        # train_answ = train_answ[condition]
        condition = ~np.isin(train['id'], bad_ids)
        overall_train_predictions_excluded = overall_train_predictions[condition]
        target_excluded = train[condition][target_name].values
        print('Exclude bad IDs from validation...')
        print('Valid data:', X_valid_excluded.shape)

    score = params['metric_function'](target_excluded, overall_train_predictions_excluded)
    print('Total score: {:.6f}'.format(score))

    return overall_train_predictions, score, model_list, imp


def predict_with_lightgbm_model(test, features, model_list):
    dtest = test[features].values
    full_preds = []
    total = 0
    for m in model_list:
        total += 1
        print('Process test model: {}'.format(total))
        preds = m.predict(dtest, num_iteration=m.best_iteration)
        full_preds.append(preds)
    preds = np.array(full_preds).mean(axis=0)
    return preds


if __name__ == '__main__':
    start_time = time.time()
    gbm_type = 'LGBM'
    params = get_params()
    target = params['target']
    id = params['id']
    metric = params['metric']

    train, test, features = read_input_data()
    print('Features: [{}] {}'.format(len(features), features))

    if 1:
        overall_train_predictions, score, model_list, importance = create_lightgbm_model(train, features, params)
        prefix = '{}_{}_{}_{:.6f}'.format(gbm_type, len(model_list), metric, score)
        save_in_file((score, model_list, importance, overall_train_predictions), MODELS_PATH + prefix + '.pklz')
    else:
        prefix = 'LGBM_5_auc_0.897168'
        score, model_list, importance, overall_train_predictions = load_from_file(MODELS_PATH + prefix + '.pklz')

    for i, c in enumerate(CLASSES):
        train[c] = overall_train_predictions[:, i]
    train[[id] + CLASSES].to_csv(SUBM_PATH + prefix + '_train.csv', index=False, float_format='%.8f')

    # SAVE OOF FEATURES
    save_in_file(overall_train_predictions, FEATURES_PATH + 'oof/' + prefix + '_train.pklz')
    overall_test_predictions = predict_with_lightgbm_model(test, features, model_list)

    # SAVE OOF FEATURES
    save_in_file(overall_test_predictions, FEATURES_PATH + 'oof/' + prefix + '_test.pklz')

    # CREATE SUBM
    for i, c in enumerate(CLASSES):
        test[c] = overall_test_predictions[:, i]
    sample = pd.read_csv(INPUT_PATH + 'submission_format.csv')
    test = pd.merge(sample[['id']], test, on='id', how='left')
    out_path = SUBM_PATH + prefix + '_ensemble_{}.csv'.format(len(features))
    out_path_1 = SUBM_PATH + 'lightgbm_ensemble.csv'
    test[[id] + CLASSES].to_csv(out_path, index=False, float_format='%.8f')
    test[[id] + CLASSES].to_csv(out_path_1, index=False, float_format='%.8f')
    print("Elapsed time overall: %s seconds" % (time.time() - start_time))

'''
Densenet121 (0.430336) + IRV2 (0.450813) + IRV2 (0.444538) + EfficientNetB4 (0.431532):
Total score: 0.396322

Densenet121 (0.430336) + IRV2 (0.450813) + IRV2 (0.444538) + EfficientNetB4 (0.431532) + DenseNet169 + New neighbours:
Total score: 0.383285

Densenet121 (0.430336) + IRV2 (0.450813) + IRV2 (0.444538) + EfficientNetB4 (0.431532) + DenseNet169 + New neighbours:
Total score: 0.379093

Pseudolabels (overfit)
Total score: 0.273755

Score iter 0: 0.379046 Time: 111.56 sec
Total score: 0.373893 

Total score: 0.378917

Total score: 0.378481
'''