# coding: utf-8
__author__ = 'ZFTurbo: https://www.drivendata.org/users/ZFTurbo/'


from catboost import Pool, CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from gbm_classifiers.a01_read_data import *


def get_kfold_split(folds_number, len_train, target, random_state):
    train_index = list(range(len_train))
    folds = StratifiedKFold(n_splits=folds_number, shuffle=True, random_state=random_state)
    ret = []
    for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train_index, target)):
        ret.append([trn_idx, val_idx])
    return ret


def get_importance(gbm, data, features):
    importance = gbm.get_feature_importance(data, thread_count=-1, fstr_type='FeatureImportance')
    imp = dict()
    for i, f in enumerate(features):
        imp[f] = importance[i]
    res = sort_dict_by_values(imp)
    return res


def create_catboost_model(train, features, params):
    import catboost as catb
    import matplotlib.pyplot as plt
    print('Catboost version: {}'.format(catb.__version__))
    target_name = params['target']
    start_time = time.time()
    train[target_name] = train[target_name]

    unique_target = np.array(sorted(train[target_name].unique()))
    print('Target length: {}: {}'.format(len(unique_target), unique_target))

    required_iterations = 13
    seed = 1921
    overall_train_predictions = np.zeros((len(train), len(unique_target)), dtype=np.float32)
    overall_importance = dict()

    model_list = []
    for iter1 in range(required_iterations):
        num_folds = random.randint(4, 5)
        learning_rate = random.choice([0.1, 0.15, 0.2])
        depth = random.choice([2, 3, 4])

        ret = get_kfold_split(num_folds, len(train), train[target_name].values, seed + iter1)
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

            # v1 (don't support GPU)
            if 0:
                early_stop = 10
                model = CatBoostClassifier(
                    loss_function="Logloss",
                    eval_metric="Logloss",
                    task_type='GPU',
                    iterations=10000,
                    learning_rate=0.3,
                    depth=3,
                    bootstrap_type='Bernoulli',
                    subsample=0.8,
                    colsample_bylevel=0.8,
                    metric_period=1,
                    od_type='Iter',
                    od_wait=early_stop,
                    random_seed=17,
                    l2_leaf_reg=10,
                    allow_writing_files=False
                )
            else:
                early_stop = 100
                model = CatBoostClassifier(
                    loss_function="MultiClass",
                    eval_metric="MultiClass",
                    task_type='CPU',
                    # task_type='GPU',
                    # devices='0',
                    iterations=10000,
                    early_stopping_rounds=early_stop,
                    learning_rate=learning_rate,
                    depth=depth,
                    random_seed=17,
                    l2_leaf_reg=10,
                    allow_writing_files=False
                )

            if 0:
                cat_features_names = [
                    'alias', 'region', 'epsg', 'tiff_id', 'count_in_radius_oof_1000'
                ]
                cat_features = []
                for cfn in cat_features_names:
                    cat_features.append(features.index(cfn))

                dtrain = Pool(X_train[features].values, label=y_train, cat_features=cat_features)
                dvalid = Pool(X_valid_excluded[features].values, label=y_valid_excluded, cat_features=cat_features)
            else:
                dtrain = Pool(X_train[features].values, label=y_train)
                dvalid = Pool(X_valid_excluded[features].values, label=y_valid_excluded)

            gbm = model.fit(dtrain, eval_set=dvalid, use_best_model=True, verbose=10)
            model_list.append(gbm)

            imp = get_importance(gbm, Pool(X_valid[features].values), features)
            print('Importance: {}'.format(imp[:100]))
            for i in imp:
                if i[0] in overall_importance:
                    overall_importance[i[0]] += i[1] / num_folds
                else:
                    overall_importance[i[0]] = i[1] / num_folds

            print('Best iter: {}'.format(gbm.get_best_iteration()))
            pred = gbm.predict_proba(X_valid[features].values)
            print(pred.shape)
            full_single_preds[valid_index] += pred.copy()

            pred = gbm.predict_proba(X_valid_excluded[features].values)
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


def predict_with_catboost_model(test, features, model_list):
    full_preds = []
    for m in model_list:
        preds = m.predict_proba(test[features].values)
        full_preds.append(preds)
    preds = np.array(full_preds).mean(axis=0)
    return preds


if __name__ == '__main__':
    start_time = time.time()
    gbm_type = 'CatB'
    params = get_params()
    target = 'target'
    id = 'id'
    metric = 'log_loss'

    train, test, features = read_input_data()
    print('Features: [{}] {}'.format(len(features), features))

    if 1:
        overall_train_predictions, score, model_list, importance = create_catboost_model(train, features, params)
        prefix = '{}_{}_{}_{:.6f}'.format(gbm_type, len(model_list), metric, score)
        save_in_file((score, model_list, importance, overall_train_predictions), MODELS_PATH + prefix + '.pklz')
    else:
        prefix = 'CatB_3_log_loss_0.692644'
        score, model_list, importance, overall_train_predictions = load_from_file(MODELS_PATH + prefix + '.pklz')

    for i, c in enumerate(CLASSES):
        train[c] = overall_train_predictions[:, i]
    train[[id] + CLASSES].to_csv(SUBM_PATH + prefix + '_train.csv', index=False, float_format='%.8f')

    overall_test_predictions = predict_with_catboost_model(test, features, model_list)

    # SAVE OOF FEATURES
    save_in_file(overall_train_predictions, FEATURES_PATH + 'oof/' + prefix + '_train.pklz')
    save_in_file(overall_test_predictions, FEATURES_PATH + 'oof/' + prefix + '_test.pklz')

    # CREATE SUBM
    for i, c in enumerate(CLASSES):
        test[c] = overall_test_predictions[:, i]
    sample = pd.read_csv(INPUT_PATH + 'submission_format.csv')
    test = pd.merge(sample[['id']], test, on='id', how='left')
    out_path = SUBM_PATH + prefix + '_ensemble_{}.csv'.format(len(features))
    out_path_1 = SUBM_PATH + 'catboost_ensemble.csv'
    test[[id] + CLASSES].to_csv(out_path, index=False, float_format='%.8f')
    test[[id] + CLASSES].to_csv(out_path_1, index=False, float_format='%.8f')
    print("Elapsed time overall: %s seconds" % (time.time() - start_time))


'''
LS: 0.692644 (All) LS: 0.965503 (Verified) LB: 0.9303

Densenet121:
Score iter 0: 0.423462 Time: 138.04 sec
Score iter 1: 0.425525 Time: 246.27 sec
Score iter 2: 0.425720 Time: 352.89 sec
Total score: 0.420820 LB: 0.3809

Densenet121 (0.430336) + IRV2 (0.450813):
Score iter 0: 0.405551 Time: 160.70 sec
Score iter 1: 0.406967 Time: 303.15 sec
Score iter 2: 0.404649 Time: 429.19 sec
Total score: 0.400706 LB: 0.3727

Densenet121 (0.430336) + IRV2 (0.450813) + IRV2 (0.444538):
Total score: 0.395695 LB: 0.3700

Densenet121 (0.430336) + IRV2 (0.450813) + IRV2 (0.444538) + EfficientNetB4 (0.431532):
Total score: 0.387795

Score iter 0: 0.396201 Time: 200.89 sec
Score iter 0: 0.394116 Time: 242.62 sec
Score iter 0: 0.392068 Time: 227.87 sec
Score iter 0: 0.387600 Time: 227.87 sec
Score iter 0: 0.387933 Time: 294.76 sec

Total score: 0.377452

Score iter 0: 0.387025 Time: 213.91 sec

Densenet121 (0.430336) + IRV2 (0.450813) + IRV2 (0.444538) + EfficientNetB4 (0.431532) + DenseNet169 + New neighbours:
Total score: 0.375957

Score iter 0: 0.385900 Time: 223.98 sec
Densenet121 (0.430336) + IRV2 (0.450813) + IRV2 (0.444538) + EfficientNetB4 (0.431532) + DenseNet169 + Pseudo v1 + New neighbours:
Total score: 0.375543

Score iter 0: 0.277752 Time: 276.90 sec - with pseudolabels (probably overfit a lot)
Total score: 0.270457 - looks like overfit LB: 0.3809

Score iter 0: 0.383663 Time: 197.85 sec - OOF neighbours (radius 1000)
Score iter 0: 0.381085 Time: 210.45 sec - OOF neighbours (radius 1000 + closest 10)
Score iter 0: 0.381755 Time: 487.92 sec - OOF neighbours + CAT features (radius 1000 + closest 10)
Score iter 0: 0.382935 Time: 67.25 sec - GPU
Score iter 0: 0.384348 Time: 96.70 sec - GPU + cat_features

Total score: 0.374865

+ ResNet34
Score iter 0: 0.386545 Time: 261.77 sec
Total score: 0.378113

Score iter 0: 0.385847 Time: 67.00 sec
Total score: 0.377802

Score iter 0: 0.388470 Time: 59.17 sec
Total score: 0.378157
'''