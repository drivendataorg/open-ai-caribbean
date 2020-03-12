import click
import lightgbm as lgb

from utils.roofs_features_009 import *

MODEL_TYPE = 'L2A'
MODEL_VERSION = 'lgb009'
OUTPUT_NAME = f"{MODEL_TYPE}_{MODEL_VERSION}"
# List of models with 1 layer predictions
MODELS_LIST = [['A03', 'C06g', 'A10', 'A11a']]


@click.command()
@click.option('--data_folder', default='data')
def main(data_folder):
    print('Models: ', MODELS_LIST)
    train_shapefile , test_shapefile, train_features_df, test_features_df = load_data(data_folder, MODELS_LIST)

    train_shapefile['roof_type_id'] = train_shapefile.roof_material.astype("category").cat.codes

    folds = pd.read_csv(f'{data_folder}/8Kfolds_201910302225.csv')
    folds = folds[folds.id.isin(train_shapefile.index.tolist())]

    additional_train_features_df, additional_test_features_df, train_shapefile, test_shapefile =\
        create_additional_features(folds, train_shapefile, test_shapefile)
    # Combine field level additional features and predictions from 1st level
    train_val_data = pd.concat((additional_train_features_df, train_features_df), axis=1)
    columns = train_val_data.columns
    train_val_data['roof_type_id'] = train_shapefile.roof_type_id
    print(train_val_data.head(3))

    test_data = pd.concat((additional_test_features_df, test_features_df), axis=1)

    print(test_data.head(3))

    # text to categorical
    name_to_id = {'borde_rural': 0, 'borde_soacha': 1, 'mixco_1_and_ebenezer': 2, 'mixco_3': 3, 'castries': 4,
                 'dennery': 5, 'gros_islet': 6}
    train_val_data['name'] = train_val_data['name'].map(lambda x: name_to_id[x])
    test_data['name'] = test_data['name'].map(lambda x: name_to_id[x])

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': 5,
        'metric': {'multi_logloss'},
        'num_threads': 16,
        'num_iterations': 5000,
        # Core parameters
        'learning_rate': 0.01,
        # Learning control parameters
        'max_depth': 4,
        'feature_fraction': 0.8,  # colsample_bytree
        'bagging_freq': 1,
        'bagging_fraction': 0.7,  # subsample
        'num_leaves': 16,
        'min_data_in_leaf': 20,

        'verbosity': -1,
    }

    all_valid_y = []
    all_valid_field_ids = []
    all_predicts = []
    all_valid_predicts = []
    for fold in sorted(folds.fold.unique()):
        train_field_ids_i = folds.loc[folds.fold != fold, 'id'].values
        val_field_ids_i = folds.loc[folds.fold == fold, 'id'].values

        X_train, X_valid = train_val_data.loc[train_field_ids_i, columns].values, train_val_data.loc[
            val_field_ids_i, columns].values
        y_train, y_valid = train_val_data.loc[train_field_ids_i, 'roof_type_id'].values, train_val_data.loc[
            val_field_ids_i, 'roof_type_id'].values

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid)
        model_lgb = lgb.train(params, train_data, valid_sets=[valid_data],
                              early_stopping_rounds=50, verbose_eval=False, categorical_feature=[0])

        print(f"Fold - {fold} - {model_lgb.best_score['valid_0']['multi_logloss']:.4}@{model_lgb.best_iteration}")
        ypred = model_lgb.predict(test_data, model_lgb.best_iteration)
        y_valid_pred = model_lgb.predict(X_valid, model_lgb.best_iteration)

        all_valid_field_ids.extend(val_field_ids_i)
        all_predicts.append(ypred)
        all_valid_y.extend(y_valid)
        all_valid_predicts.append(y_valid_pred)

    calc_local_loss(all_valid_predicts, all_valid_y)
    save_final_predictions(OUTPUT_NAME, all_predicts, test_shapefile, all_valid_predicts, all_valid_field_ids)


if __name__ == '__main__':
    main()
