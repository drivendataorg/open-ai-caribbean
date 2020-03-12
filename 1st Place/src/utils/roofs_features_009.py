import os

import numpy as np
import geopandas as gpd
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import log_loss
from tqdm import tqdm

OUTPUT_PATH = './predictions'
TRAIN_FEATURES_TEMPLATE = '{predictions_folder}/L1A_{model_name}_folds.csv.gz'
TEST_FEATURES_TEMPLATE = '{predictions_folder}/L1A_{model_name}_test.csv.gz'


def load_data(data_folder, models_list_list, predictions_folder='predictions'):
    """
    Loads first level train ans test features from files and blends them.
    Loads train and test geojson data about polygons/roofs
    """
    print('Loading 1st level model features...')
    train_shapefile = gpd.read_file(f"{data_folder}/train.geojson")
    train_shapefile.set_index('id', inplace=True)
    test_shapefile = gpd.read_file(f"{data_folder}/test.geojson")
    test_shapefile.set_index('id', inplace=True)

    train_features = []
    for i, models_list in enumerate(models_list_list):
        train_features_df_i = pd.read_csv(TRAIN_FEATURES_TEMPLATE.format(predictions_folder=predictions_folder, model_name=models_list_list[i][0]), index_col=0)
        for model_name in models_list[1:]:
            train_features_df_i += pd.read_csv(TRAIN_FEATURES_TEMPLATE.format(predictions_folder=predictions_folder, model_name=model_name), index_col=0)
        train_features_df_i.columns = [column + '_' + str(i) for column in train_features_df_i.columns]
        train_features.append(train_features_df_i)
    train_features_df = pd.concat(train_features, axis=1)
    print('Train data shape - ', train_features_df.shape)
    print(train_features_df.head())

    test_features = []
    for i, models_list in enumerate(models_list_list):
        test_features_df_i = pd.read_csv(TEST_FEATURES_TEMPLATE.format(predictions_folder=predictions_folder, model_name=models_list_list[i][0]), index_col=0)
        for model_name in models_list[1:]:
            test_features_df_i += pd.read_csv(TEST_FEATURES_TEMPLATE.format(predictions_folder=predictions_folder, model_name=model_name), index_col=0)
        test_features_df_i.columns = [column + '_' + str(i) for column in test_features_df_i.columns]
        test_features.append(test_features_df_i)
    test_features_df = pd.concat(test_features, axis=1)
    print('Test data shape - ', test_features_df.shape)
    print(test_features_df.head())

    return train_shapefile, test_shapefile, train_features_df, test_features_df


def add_centroid_dist_feature(train_df, test_df, num_classes=5):
    """
    Calculates the number of closest neighbor roofs(max 9) per class with distance coefficient - (0.001 / dist) ** 2,
    where dist is the distance between a roof centroid in question and centroids of neighbor roofs (5 features).
    """
    print('Start creating a cKDTree...')
    nB = np.array(list(zip(train_df.geometry.centroid.x.values, train_df.geometry.centroid.y.values)))
    btree = cKDTree(nB, compact_nodes=True, balanced_tree=True)
    print('Tree has been built')

    for col in [f'neighbour_{col}' for col in list(range(0, num_classes))]:
        test_df[col] = 0.0

    for i, row in tqdm(test_df.iterrows()):
        scores = {x: 0 for x in [f'neighbour_{col}' for col in list(range(0, num_classes))]}
        pt = row['centroid']
        i_start = 0
        k = 9
        dists, idxs = btree.query(np.array([pt.x, pt.y]), k=k)
        for dist, idx in zip(dists[i_start:k], idxs[i_start:k]):
            roof_id = int(train_df.loc[train_df.index[idx], 'roof_type_id'])
            scores[f'neighbour_{roof_id}'] += (0.001 / dist) ** 2.0
        for col, score in scores.items():
            test_df.loc[i, col] = score
    return test_df.loc[:, [f'neighbour_{col}' for col in list(range(0, num_classes))]]


def get_angle(p0, p1, p2):
    """
    Computes angle (in degrees) for p0p1p2 corner
    Inputs:
        p0,p1,p2 - points in the form of [x,y]
    """
    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)

    angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
    return np.degrees(angle)


def get_min_angle(polygon):
    """
    Calculates minimum angle in the polygon
    """
    angle_list = []
    points = list(polygon.convex_hull.exterior.coords)
    points = [points[-2]] + points
    for i in range(1, len(points) - 1):
        angle = get_angle(points[i-1], points[i], points[i+1])
        angle_list.append(angle)
    return min(angle_list)


def get_num_points(polygon):
    """
    Calculates the number of points in the polygon
    """
    points = list(polygon.convex_hull.exterior.coords)
    return len(points)


def create_additional_features(folds, train_shapefile, test_shapefile, num_classes=5):
    """
    Calculates all the features for both train and test data
    """
    # Find the smallest angle in roof polygons
    train_shapefile['min_angle'] = train_shapefile.geometry.apply(lambda x: get_min_angle(x))
    test_shapefile['min_angle'] = test_shapefile.geometry.apply(lambda x: get_min_angle(x))

    # Calculate number of points in roof polygons
    train_shapefile['num_points'] = train_shapefile.geometry.apply(lambda x: get_num_points(x))
    test_shapefile['num_points'] = test_shapefile.geometry.apply(lambda x: get_num_points(x))

    # Create centroid features
    train_shapefile['centroid'] = train_shapefile.geometry.centroid
    test_shapefile['centroid'] = test_shapefile.geometry.centroid

    # Create area features
    train_shapefile['area'] = train_shapefile.geometry.area
    test_shapefile['area'] = test_shapefile.geometry.area

    # Add latitude and longitude features
    train_shapefile['lat'] = train_shapefile.geometry.centroid.x.values
    train_shapefile['long'] = train_shapefile.geometry.centroid.y.values
    test_shapefile['lat'] = test_shapefile.geometry.centroid.x.values
    test_shapefile['long'] = test_shapefile.geometry.centroid.y.values

    print("Creating additional distance/intersect features ...")
    for i in [f'neighbour_{col}' for col in list(range(0, num_classes))]:
        train_shapefile[i] = 0.0
    for fold in sorted(folds.fold.unique()):
        print(f'Fold - {fold}...')
        train_field_ids_i = folds.loc[folds.fold != fold, 'id']
        val_field_ids_i = folds.loc[folds.fold == fold, 'id']
        features = add_centroid_dist_feature(train_shapefile.loc[train_field_ids_i, :],
                                             train_shapefile.loc[val_field_ids_i, :])
        train_shapefile.loc[features.index, [f'neighbour_{col}' for col in list(range(0, num_classes))]] = features

        print(train_shapefile.head(10))

    for i in [f'neighbour_{col}' for col in list(range(0, num_classes))]:
        test_shapefile[i] = 0.0
    features = add_centroid_dist_feature(train_shapefile, test_shapefile)
    test_shapefile.loc[features.index, [f'neighbour_{col}' for col in list(range(0, num_classes))]] = features

    columns = ['name', 'num_points', 'min_angle', 'area', 'lat', 'long',  *[f'neighbour_{col}' for col in list(range(0, num_classes))]]
    additional_train_features_df = train_shapefile.loc[:, columns]
    additional_train_features_df = pd.DataFrame(additional_train_features_df)
    print(additional_train_features_df.head())

    additional_test_features_df = test_shapefile.loc[:, columns].copy()
    additional_test_features_df = pd.DataFrame(additional_test_features_df)
    print(additional_test_features_df.head())

    return additional_train_features_df, additional_test_features_df, train_shapefile, test_shapefile


def calc_local_loss(all_valid_predicts, all_valid_y):
    """
    Calculates and prints local validation loss
    """
    log_loss_value = log_loss(all_valid_y, np.vstack(all_valid_predicts))
    print(f'CV logloss - {log_loss_value}')


def save_final_predictions(output_name, all_predicts, test_shapefile, all_valid_predicts, all_valid_field_ids):
    """
    Saves validation and average test prediction into files
    """
    valid_preds_df = pd.DataFrame(np.vstack(all_valid_predicts), index=all_valid_field_ids,
                                  columns=['concrete_cement','healthy_metal','incomplete','irregular_metal','other'])
    valid_preds_df.index.name = 'id'
    filepath = os.path.join(OUTPUT_PATH, output_name + '_folds.csv.gz')
    print(f"Saving VALIDATION predictions: {filepath}")
    valid_preds_df.to_csv(filepath, index=True)

    result = np.zeros_like(all_predicts[0])
    for predict in all_predicts:
        result += predict
    result = result / len(all_predicts)
    test_preds_df = pd.DataFrame(result, index=test_shapefile.index.tolist())
    test_preds_df.columns = ['concrete_cement', 'healthy_metal', 'incomplete', 'irregular_metal', 'other']
    test_preds_df.index.name = 'id'
    filepath = os.path.join(OUTPUT_PATH, output_name + '_test.csv.gz')
    print(f"Saving TEST predictions: {filepath}")
    test_preds_df.to_csv(filepath, index=True)
