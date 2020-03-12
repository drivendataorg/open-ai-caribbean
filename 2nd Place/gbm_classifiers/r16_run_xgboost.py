# coding: utf-8
__author__ = 'ZFTurbo: https://www.drivendata.org/users/ZFTurbo/'


from operator import itemgetter
from gbm_classifiers.a01_read_data import *


def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()


def get_importance(gbm, features):
    create_feature_map(features)
    '''
    ‘weight’ - the number of times a feature is used to split the data across all trees.
    ‘gain’ - the average gain of the feature when it is used in trees
    ‘cover’ - the average coverage of the feature when it is used in trees
    '''
    importance = gbm.get_score(fmap='xgb.fmap', importance_type='weight')
    importance = sorted(importance.items(), key=itemgetter(1), reverse=True)
    return importance


def get_kfold_split(folds_number, len_train, target, random_state):
    train_index = list(range(len_train))
    folds = StratifiedKFold(n_splits=folds_number, shuffle=True, random_state=random_state)
    ret = []
    for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train_index, target)):
        ret.append([trn_idx, val_idx])
    return ret


def create_xgboost_model(train, features, params):
    import xgboost as xgb
    import matplotlib.pyplot as plt
    print('XGBoost version: {}'.format(xgb.__version__))
    target_name = params['target']
    start_time = time.time()

    unique_target = np.array(sorted(train[target_name].unique()))
    print('Target length: {}: {}'.format(len(unique_target), unique_target))

    required_iterations = 13
    overall_train_predictions = np.zeros((len(train), len(unique_target)), dtype=np.float32)
    overall_importance = dict()

    model_list = []
    for iter1 in range(required_iterations):
        num_folds = 5
        max_depth = random.choice([3, 4, 5])
        eta = random.choice([0.08, 0.1, 0.2])

        subsample = random.choice([0.7, 0.8, 0.9, 0.95])
        colsample_bytree = random.choice([0.7, 0.8, 0.9, 0.95])
        eval_metric = random.choice(['mlogloss'])
        # eval_metric = random.choice(['logloss'])
        ret = get_kfold_split(num_folds, len(train), train[target_name].values, 720 + iter1)

        log_str = 'XGBoost iter {}. FOLDS: {} METRIC: {} ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(0,
                                                                                                               num_folds,
                                                                                                               eval_metric,
                                                                                                               eta,
                                                                                                               max_depth,
                                                                                                               subsample,
                                                                                                               colsample_bytree)
        print(log_str)
        params_xgb = {
            "objective": "multi:softprob",
            "num_class": 5,
            "booster": "gbtree",
            "eval_metric": eval_metric,
            "eta": eta,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "silent": 1,
            "seed": 2017 + iter1,
            "nthread": 6,
            "gamma": 0,
            "tree_method": 'exact',
            # 'gpu_id': 0,
            # "tree_method": 'gpu_hist',
            # 'updater': 'grow_gpu',
        }
        num_boost_round = 10000
        early_stopping_rounds = 50

        # print('Train shape:', train.shape)
        # print('Features:', features)

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

            dtrain = xgb.DMatrix(X_train[features].values, y_train)
            dvalid = xgb.DMatrix(X_valid_excluded[features].values, y_valid_excluded)

            watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
            gbm = xgb.train(params_xgb, dtrain, num_boost_round, evals=watchlist,
                            early_stopping_rounds=early_stopping_rounds, verbose_eval=5)
            model_list.append(gbm)

            imp = get_importance(gbm, features)
            print('Importance: {}'.format(imp[:100]))
            for i in imp:
                if i[0] in overall_importance:
                    overall_importance[i[0]] += i[1] / num_folds
                else:
                    overall_importance[i[0]] = i[1] / num_folds

            print('Best iter: {}'.format(gbm.best_iteration + 1))
            pred = gbm.predict(xgb.DMatrix(X_valid[features].values), ntree_limit=gbm.best_iteration + 1)
            full_single_preds[valid_index] += pred.copy()

            pred = gbm.predict(dvalid, ntree_limit=gbm.best_iteration + 1)
            try:
                score = params['metric_function'](y_valid_excluded, pred)
                print('Fold {} score: {:.6f}'.format(fold_num, score))
            except Exception as e:
                print('Error:', e)

        print(train[target_name].values)
        print(full_single_preds)

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


def predict_with_xgboost_model(test, features, model_list):
    import xgboost as xgb

    dtest = xgb.DMatrix(test[features].values)
    full_preds = []
    for m in model_list:
        preds = m.predict(dtest, ntree_limit=m.best_iteration + 1)
        full_preds.append(preds)
    preds = np.array(full_preds).mean(axis=0)
    return preds


if __name__ == '__main__':
    start_time = time.time()
    gbm_type = 'XGB'
    params = get_params()
    target = params['target']
    id = params['id']
    metric = params['metric']

    train, test, features = read_input_data()
    print('Features: [{}] {}'.format(len(features), features))

    if 1:
        overall_train_predictions, score, model_list, importance = create_xgboost_model(train, features, params)
        prefix = '{}_{}_{}_{:.6f}'.format(gbm_type, len(model_list), metric, score)
        save_in_file((score, model_list, importance, overall_train_predictions), MODELS_PATH + prefix + '.pklz')
    else:
        prefix = 'XGB_5_auc_0.891512'
        score, model_list, importance, overall_train_predictions = load_from_file(MODELS_PATH + prefix + '.pklz')

    for i, c in enumerate(CLASSES):
        train[c] = overall_train_predictions[:, i]
    train[[id] + CLASSES].to_csv(SUBM_PATH + prefix + '_train.csv', index=False, float_format='%.8f')

    overall_test_predictions = predict_with_xgboost_model(test, features, model_list)

    # SAVE OOF FEATURES
    save_in_file(overall_train_predictions, FEATURES_PATH + 'oof/' + prefix + '_train.pklz')
    save_in_file(overall_test_predictions, FEATURES_PATH + 'oof/' + prefix + '_test.pklz')

    # CREATE SUBM
    for i, c in enumerate(CLASSES):
        test[c] = overall_test_predictions[:, i]
    sample = pd.read_csv(INPUT_PATH + 'submission_format.csv')
    test = pd.merge(sample[['id']], test, on='id', how='left')
    out_path = SUBM_PATH + prefix + '_ensemble_{}.csv'.format(len(features))
    out_path_1 = SUBM_PATH + 'xgboost_ensemble.csv'
    test[[id] + CLASSES].to_csv(out_path, index=False, float_format='%.8f')
    test[[id] + CLASSES].to_csv(out_path_1, index=False, float_format='%.8f')
    print("Elapsed time overall: %s seconds" % (time.time() - start_time))


'''
Densenet121 (0.430336) + IRV2 (0.450813) + IRV2 (0.444538):
Score iter 0: 0.399968 Time: 88.66 sec
Score iter 2: 0.400587 Time: 316.98 sec
Total score: 0.394132 LB: 0.3700

Densenet121 (0.430336) + IRV2 (0.450813) + IRV2 (0.444538) + EfficientNetB4 (0.431532):
Total score: 0.386864

Densenet121 (0.430336) + IRV2 (0.450813) + IRV2 (0.444538) + EfficientNetB4 (0.431532) + DenseNet169 + New neighbours:
Total score: 0.373275

Pseudolabels (overfit)
Total score: 0.268235

Score iter 0: 0.379637 Time: 70.15 sec - GPU
Total score: 0.370199 LB: 0.3639

Total score: 0.374172

Total score: 0.373778
'''