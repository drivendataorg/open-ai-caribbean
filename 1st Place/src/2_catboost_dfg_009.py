import click
from catboost import CatBoostClassifier

from utils.roofs_features_009 import *

MODEL_TYPE = 'L2A'
MODEL_VERSION = 'cat009'
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

        model_cat = CatBoostClassifier(
            iterations=10000,
            random_seed=111,
            learning_rate=0.01,
            logging_level='Silent',
            task_type="GPU",
            devices='0',
            max_depth=5,
            l2_leaf_reg=3.0,
            bagging_temperature=1,
        )

        model_cat.fit(
            X_train, y_train,
            eval_set=(X_valid, y_valid),
            cat_features=[0])

        print(f"Fold - {fold} - {model_cat.get_best_score()['validation']['MultiClass']:.4}@{model_cat.get_best_iteration()}")
        ypred = model_cat.predict_proba(test_data, ntree_end=model_cat.get_best_iteration())
        y_valid_pred = model_cat.predict_proba(X_valid, ntree_end=model_cat.get_best_iteration())

        all_valid_field_ids.extend(val_field_ids_i)
        all_predicts.append(ypred)
        all_valid_y.extend(y_valid)
        all_valid_predicts.append(y_valid_pred)

    calc_local_loss(all_valid_predicts, all_valid_y)
    save_final_predictions(OUTPUT_NAME, all_predicts, test_shapefile, all_valid_predicts, all_valid_field_ids)


if __name__ == '__main__':
    main()
