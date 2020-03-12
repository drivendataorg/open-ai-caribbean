# coding: utf-8
__author__ = 'ZFTurbo: https://www.drivendata.org/users/ZFTurbo/'


from a00_common_functions import *
import rasterio
import pandas as pd
from pyproj import Proj
import geopandas as gpd
from pathlib import Path
import cv2
from shapely.ops import cascaded_union
from shapely.geometry import Polygon


def polygon_to_tuples(poly):
    if str(poly.type) == 'MultiPolygon':
        allparts = [p.buffer(0) for p in poly]
        poly = cascaded_union(allparts)
        if poly.geom_type == 'MultiPolygon':
            area = -1
            best_p = -1
            for p in poly:
                if p.area > area:
                    area = p.area
                    best_p = p
            poly = best_p
    x, y = poly.exterior.coords.xy
    x = list(x)
    y = list(y)
    return list(zip(x, y))


def transform_coordinates(coordinate_lists,
                          transform):
    transformed_coordinates_lists = []
    for coordinates_list in coordinate_lists:
        transformed_coordinates_list = []
        for coordinates in coordinates_list:
            transformed_coordinate = tuple(transform(coordinates[0],
                                                     coordinates[1]))
            transformed_coordinates_list.append(transformed_coordinate)
        transformed_coordinates_lists.append(transformed_coordinates_list)
    return transformed_coordinates_lists


def get_train_test_df(train_geo_paths,
                 tiff_paths,
                 data_path):
    dfs = []

    region_list = []
    for t in sorted(tiff_paths):
        region_list.append(t.split('/')[-3])
    alias_list = []
    for t in sorted(tiff_paths):
        alias_list.append(t.split('/')[-2])

    for train_path, tiff_path in zip(train_geo_paths, tiff_paths):
        print('Go for: {}'.format(train_path))
        print('Go for: {}'.format(tiff_path))
        if str(train_path) == 'nan':
            continue
        tiff_path = str(tiff_path)
        a = alias_list.index(tiff_path.split('/')[-2])
        r = region_list.index(tiff_path.split('/')[-3])
        df = gpd.read_file(data_path + train_path)
        df['tiff_path'] = tiff_path
        df['alias'] = a
        df['region'] = r
        with rasterio.open(data_path + tiff_path) as tiff:
            epsg = tiff.crs.to_epsg()
            df['epsg'] = epsg
        dfs.append(df)

    df = pd.concat(dfs)
    print(df)

    return df


def process_single_row(train_df, index, type):
    delta = 100
    sample = train_df.iloc[index]
    polygon = sample.geometry
    if 'MULTI' in str(polygon):
        print('Some multi here', sample.id)

    tuples = polygon_to_tuples(polygon)
    proj = Proj(init='epsg:{}'.format(sample.epsg))
    tiff_path = data_path + sample.tiff_path
    coordinates = transform_coordinates([tuples], proj)[0]

    tiff = rasterio.open(tiff_path)
    # print('Tiff size:', tiff.width, tiff.height)
    pixels = [tiff.index(*coord) for coord in coordinates]
    xmin, xmax, ymin, ymax = 1000000000, -10000000000, 1000000000, -10000000000
    for w, h in pixels:
        if w > xmax:
            xmax = w
        if w < xmin:
            xmin = w
        if h > ymax:
            ymax = h
        if h < ymin:
            ymin = h
    if xmin < delta:
        print('Too small xmin')
    if ymin < delta:
        print('Too small ymin')
    if xmax > tiff.width - delta:
        print('Too big xmax')
    if ymax > tiff.height - delta:
        print('Too big ymax')

    center_x = (xmax + xmin) // 2
    center_y = (ymax + ymin) // 2
    area = 10000000000 * polygon.area
    return center_x, center_y, area


def add_data_to_table(table, tiff_paths):
    new_data = []
    for i in range(len(table)):
        data = process_single_row(table, i, 'train')
        new_data.append(data)
    new_data = np.array(new_data)
    print(new_data.shape)
    table['center_x'] = new_data[:, 0]
    table['center_y'] = new_data[:, 1]
    table['area'] = new_data[:, 2]

    table['tiff_id'] = -1
    for i, t in enumerate(sorted(tiff_paths)):
        table.loc[table['tiff_path'] == t, 'tiff_id'] = i

    table.drop('geometry', axis=1, inplace=True)
    table.drop('tiff_path', axis=1, inplace=True)
    return table


def create_feature_tables_init():
    sub_df = pd.read_csv(INPUT_PATH + 'submission_format.csv')
    trn_df = pd.read_csv(INPUT_PATH + 'train_labels.csv')
    met_df = pd.read_csv(INPUT_PATH + 'metadata.csv')

    tiff_paths = list(met_df.image.values)
    train_geo_paths = list(met_df.train.values)
    test_geo_paths = list(met_df.test.values)
    print(len(tiff_paths), len(train_geo_paths), len(test_geo_paths))
    print(tiff_paths)
    print(train_geo_paths)
    print(test_geo_paths)

    train_df = get_train_test_df(train_geo_paths,
                                 tiff_paths,
                                 data_path)
    test_df = get_train_test_df(test_geo_paths,
                                tiff_paths,
                                data_path)

    if 1:
        test_df = add_data_to_table(test_df, tiff_paths)
        test_df.to_csv(FEATURES_PATH + 'test_neigbours.csv', index=False)

    if 1:
        train_df = add_data_to_table(train_df, tiff_paths)
        train_df.to_csv(FEATURES_PATH + 'train_neigbours.csv', index=False)


def find_neigbours(nfind):
    from sklearn.metrics import pairwise_distances
    num_folds = 5
    train = pd.read_csv(FEATURES_PATH + 'train_neigbours.csv')
    test = pd.read_csv(FEATURES_PATH + 'test_neigbours.csv')

    if 1:
        # First found OOF for train/valid

        train_ids_all, train_answ_all, valid_ids_all, valid_answ_all = get_kfold_split(num_folds)
        oof_table = []
        for fold_number in range(num_folds):
            print('Go for fold: {}'.format(fold_number))
            train_ids = train_ids_all[fold_number]
            train_answ = train_answ_all[fold_number]
            valid_ids = valid_ids_all[fold_number]
            valid_answ = valid_answ_all[fold_number]

            train_fold = train[train['id'].isin(train_ids)].copy()
            valid_fold = train[train['id'].isin(valid_ids)].copy()
            for tiff_id in train['tiff_id'].unique():
                print('Go for tiff_id: {}'.format(tiff_id))
                train_single_tiff = train_fold[train_fold['tiff_id'] == tiff_id].copy()
                valid_single_tiff = valid_fold[valid_fold['tiff_id'] == tiff_id].copy()
                train_single_tiff.reset_index(drop=True, inplace=True)
                valid_single_tiff.reset_index(drop=True, inplace=True)

                train_points = train_single_tiff[['center_x', 'center_y']].values
                valid_points = valid_single_tiff[['center_x', 'center_y']].values
                distances = pairwise_distances(valid_points, train_points)
                print(distances.shape)
                positions = distances.argsort(axis=1)

                store_info = []
                for i in range(len(valid_points)):
                    single_n = []
                    for j in range(nfind):
                        pos_in_train = positions[i, j]
                        # print(distances[i, pos_in_train], train_points[pos_in_train])
                        # print(train_single_tiff.loc[pos_in_train].values)
                        table_line = train_single_tiff.loc[pos_in_train].values
                        dist = distances[i, pos_in_train]
                        clss = CLASSES.index(table_line[1])
                        center_x, center_y = train_points[pos_in_train]
                        area = table_line[8]
                        a = [dist, clss, center_x, center_y, area]
                        single_n += a
                    store_info.append(single_n)
                store_info = np.array(store_info)

                features = []
                for j in range(nfind):
                    features += ['nn_dist_{}'.format(j), 'nn_clss_{}'.format(j), 'nn_center_x_{}'.format(j), 'nn_center_y_{}'.format(j), 'nn_area_{}'.format(j)]

                print(store_info.shape)
                for i, f in enumerate(features):
                    valid_single_tiff[f] = store_info[:, i]
                oof_table.append(valid_single_tiff)

        oof_table = pd.concat((oof_table), axis=0)
        oof_table[['id'] + features].to_csv(FEATURES_PATH + 'neighbours_feat_{}_train.csv'.format(nfind), index=False)

    if 1:
        # test table
        train_fold = train.copy()
        valid_fold = test.copy()
        oof_table = []
        for tiff_id in train['tiff_id'].unique():
            print('Go for tiff_id: {}'.format(tiff_id))
            train_single_tiff = train_fold[train_fold['tiff_id'] == tiff_id].copy()
            valid_single_tiff = valid_fold[valid_fold['tiff_id'] == tiff_id].copy()
            train_single_tiff.reset_index(drop=True, inplace=True)
            valid_single_tiff.reset_index(drop=True, inplace=True)
            if len(valid_single_tiff) == 0:
                print('No entries')
                continue

            train_points = train_single_tiff[['center_x', 'center_y']].values
            valid_points = valid_single_tiff[['center_x', 'center_y']].values
            distances = pairwise_distances(valid_points, train_points)
            print(distances.shape)
            positions = distances.argsort(axis=1)

            store_info = []
            for i in range(len(valid_points)):
                single_n = []
                for j in range(nfind):
                    pos_in_train = positions[i, j]
                    # print(distances[i, pos_in_train], train_points[pos_in_train])
                    # print(train_single_tiff.loc[pos_in_train].values)
                    table_line = train_single_tiff.loc[pos_in_train].values
                    dist = distances[i, pos_in_train]
                    clss = CLASSES.index(table_line[1])
                    center_x, center_y = train_points[pos_in_train]
                    area = table_line[8]
                    a = [dist, clss, center_x, center_y, area]
                    single_n += a
                store_info.append(single_n)
            store_info = np.array(store_info)

            features = []
            for j in range(nfind):
                features += ['nn_dist_{}'.format(j), 'nn_clss_{}'.format(j), 'nn_center_x_{}'.format(j), 'nn_center_y_{}'.format(j), 'nn_area_{}'.format(j)]

            print(store_info.shape)
            for i, f in enumerate(features):
                valid_single_tiff[f] = store_info[:, i]
            oof_table.append(valid_single_tiff)

        oof_table = pd.concat((oof_table), axis=0)
        oof_table[['id'] + features].to_csv(FEATURES_PATH + 'neighbours_feat_{}_test.csv'.format(nfind), index=False)


def class_distribution_for_neighbours(nfind):
    from sklearn.metrics import pairwise_distances
    num_folds = 5
    train = pd.read_csv(FEATURES_PATH + 'train_neigbours.csv')
    test = pd.read_csv(FEATURES_PATH + 'test_neigbours.csv')

    if 1:
        # First found OOF for train/valid

        train_ids_all, train_answ_all, valid_ids_all, valid_answ_all = get_kfold_split(num_folds)
        oof_table = []
        for fold_number in range(num_folds):
            print('Go for fold: {}'.format(fold_number))
            train_ids = train_ids_all[fold_number]
            train_answ = train_answ_all[fold_number]
            valid_ids = valid_ids_all[fold_number]
            valid_answ = valid_answ_all[fold_number]

            train_fold = train[train['id'].isin(train_ids)].copy()
            valid_fold = train[train['id'].isin(valid_ids)].copy()
            for tiff_id in train['tiff_id'].unique():
                print('Go for tiff_id: {}'.format(tiff_id))
                train_single_tiff = train_fold[train_fold['tiff_id'] == tiff_id].copy()
                valid_single_tiff = valid_fold[valid_fold['tiff_id'] == tiff_id].copy()
                train_single_tiff.reset_index(drop=True, inplace=True)
                valid_single_tiff.reset_index(drop=True, inplace=True)

                train_points = train_single_tiff[['center_x', 'center_y']].values
                valid_points = valid_single_tiff[['center_x', 'center_y']].values
                distances = pairwise_distances(valid_points, train_points)
                print(distances.shape)
                positions = distances.argsort(axis=1)

                store_info = []
                for i in range(len(valid_points)):
                    single_n = [0] * len(CLASSES)
                    for j in range(nfind):
                        pos_in_train = positions[i, j]
                        table_line = train_single_tiff.loc[pos_in_train].values
                        clss = CLASSES.index(table_line[1])
                        single_n[clss] += 1
                    for j in range(len(single_n)):
                        single_n[j] /= nfind
                    store_info.append(single_n)
                store_info = np.array(store_info)

                features = []
                for j in range(len(CLASSES)):
                    features += ['neighbour_avg_{}_{}'.format(nfind, j)]

                print(store_info.shape)
                for i, f in enumerate(features):
                    valid_single_tiff[f] = store_info[:, i]
                oof_table.append(valid_single_tiff)

        oof_table = pd.concat((oof_table), axis=0)
        oof_table[['id'] + features].to_csv(FEATURES_PATH + 'neighbours_clss_distribution_{}_train.csv'.format(nfind), index=False)

    if 1:
        # test table
        train_fold = train.copy()
        valid_fold = test.copy()
        oof_table = []
        for tiff_id in train['tiff_id'].unique():
            print('Go for tiff_id: {}'.format(tiff_id))
            train_single_tiff = train_fold[train_fold['tiff_id'] == tiff_id].copy()
            valid_single_tiff = valid_fold[valid_fold['tiff_id'] == tiff_id].copy()
            train_single_tiff.reset_index(drop=True, inplace=True)
            valid_single_tiff.reset_index(drop=True, inplace=True)
            if len(valid_single_tiff) == 0:
                print('No entries')
                continue

            train_points = train_single_tiff[['center_x', 'center_y']].values
            valid_points = valid_single_tiff[['center_x', 'center_y']].values
            distances = pairwise_distances(valid_points, train_points)
            print(distances.shape)
            positions = distances.argsort(axis=1)

            store_info = []
            for i in range(len(valid_points)):
                single_n = [0] * len(CLASSES)
                for j in range(nfind):
                    pos_in_train = positions[i, j]
                    table_line = train_single_tiff.loc[pos_in_train].values
                    clss = CLASSES.index(table_line[1])
                    single_n[clss] += 1
                for j in range(len(single_n)):
                    single_n[j] /= nfind
                store_info.append(single_n)
            store_info = np.array(store_info)

            features = []
            for j in range(len(CLASSES)):
                features += ['neighbour_avg_{}_{}'.format(nfind, j)]

            print(store_info.shape)
            for i, f in enumerate(features):
                valid_single_tiff[f] = store_info[:, i]
            oof_table.append(valid_single_tiff)

        oof_table = pd.concat((oof_table), axis=0)
        oof_table[['id'] + features].to_csv(FEATURES_PATH + 'neighbours_clss_distribution_{}_test.csv'.format(nfind), index=False)


def class_distribution_in_radius_for_neighbours(radius_find):
    from sklearn.metrics import pairwise_distances
    num_folds = 10
    train = pd.read_csv(FEATURES_PATH + 'train_neigbours.csv')
    test = pd.read_csv(FEATURES_PATH + 'test_neigbours.csv')

    if 1:
        # First found OOF for train/valid

        train_ids_all, train_answ_all, valid_ids_all, valid_answ_all = get_kfold_split(num_folds)
        oof_table = []
        for fold_number in range(num_folds):
            print('Go for fold: {}'.format(fold_number))
            train_ids = train_ids_all[fold_number]
            train_answ = train_answ_all[fold_number]
            valid_ids = valid_ids_all[fold_number]
            valid_answ = valid_answ_all[fold_number]

            train_fold = train[train['id'].isin(train_ids)].copy()
            valid_fold = train[train['id'].isin(valid_ids)].copy()
            for tiff_id in train['tiff_id'].unique():
                print('Go for tiff_id: {}'.format(tiff_id))
                train_single_tiff = train_fold[train_fold['tiff_id'] == tiff_id].copy()
                valid_single_tiff = valid_fold[valid_fold['tiff_id'] == tiff_id].copy()
                train_single_tiff.reset_index(drop=True, inplace=True)
                valid_single_tiff.reset_index(drop=True, inplace=True)

                train_points = train_single_tiff[['center_x', 'center_y']].values
                valid_points = valid_single_tiff[['center_x', 'center_y']].values
                distances = pairwise_distances(valid_points, train_points)
                print(distances.shape)
                positions = distances.argsort(axis=1)

                store_info = []
                for i in range(len(valid_points)):
                    single_n = [0]*len(CLASSES)
                    count_in_radius = 0
                    for j in range(positions.shape[1]):
                        pos_in_train = positions[i, j]
                        # print(distances[i, pos_in_train], train_points[pos_in_train])
                        # print(train_single_tiff.loc[pos_in_train].values)
                        table_line = train_single_tiff.loc[pos_in_train].values
                        dist = distances[i, pos_in_train]
                        if dist > radius_find:
                            break
                        clss = CLASSES.index(table_line[1])
                        single_n[clss] += 1
                        count_in_radius += 1
                    if count_in_radius > 0:
                        for j in range(len(single_n)):
                            single_n[j] /= count_in_radius
                    store_info.append([count_in_radius] + single_n)
                store_info = np.array(store_info)

                features = ['count_in_radius_{}'.format(radius_find)]
                for j in range(len(CLASSES)):
                    features += ['neighbour_in_radius_{}_{}'.format(radius_find, j)]

                print(store_info.shape)
                for i, f in enumerate(features):
                    valid_single_tiff[f] = store_info[:, i]
                oof_table.append(valid_single_tiff)

        oof_table = pd.concat((oof_table), axis=0)
        oof_table['count_in_radius_{}'.format(radius_find)] = oof_table['count_in_radius_{}'.format(radius_find)].astype(np.int32)
        oof_table[['id'] + features].to_csv(FEATURES_PATH + 'neighbours_clss_distribution_radius_{}_train.csv'.format(radius_find), index=False)

    if 1:
        # test table
        train_fold = train.copy()
        valid_fold = test.copy()
        oof_table = []
        for tiff_id in train['tiff_id'].unique():
            print('Go for tiff_id: {}'.format(tiff_id))
            train_single_tiff = train_fold[train_fold['tiff_id'] == tiff_id].copy()
            valid_single_tiff = valid_fold[valid_fold['tiff_id'] == tiff_id].copy()
            train_single_tiff.reset_index(drop=True, inplace=True)
            valid_single_tiff.reset_index(drop=True, inplace=True)
            if len(valid_single_tiff) == 0:
                print('No entries')
                continue

            train_points = train_single_tiff[['center_x', 'center_y']].values
            valid_points = valid_single_tiff[['center_x', 'center_y']].values
            distances = pairwise_distances(valid_points, train_points)
            print(distances.shape)
            positions = distances.argsort(axis=1)

            store_info = []
            for i in range(len(valid_points)):
                single_n = [0] * len(CLASSES)
                count_in_radius = 0
                for j in range(positions.shape[1]):
                    pos_in_train = positions[i, j]
                    # print(distances[i, pos_in_train], train_points[pos_in_train])
                    # print(train_single_tiff.loc[pos_in_train].values)
                    table_line = train_single_tiff.loc[pos_in_train].values
                    dist = distances[i, pos_in_train]
                    if dist > radius_find:
                        break
                    clss = CLASSES.index(table_line[1])
                    single_n[clss] += 1
                    count_in_radius += 1
                if count_in_radius > 0:
                    for j in range(len(single_n)):
                        single_n[j] /= count_in_radius
                store_info.append([count_in_radius] + single_n)
            store_info = np.array(store_info)

            features = ['count_in_radius_{}'.format(radius_find)]
            for j in range(len(CLASSES)):
                features += ['neighbour_in_radius_{}_{}'.format(radius_find, j)]

            print(store_info.shape)
            for i, f in enumerate(features):
                valid_single_tiff[f] = store_info[:, i]
            oof_table.append(valid_single_tiff)

        oof_table = pd.concat((oof_table), axis=0)
        oof_table['count_in_radius_{}'.format(radius_find)] = oof_table['count_in_radius_{}'.format(radius_find)].astype(np.int32)
        oof_table[['id'] + features].to_csv(FEATURES_PATH + 'neighbours_clss_distribution_radius_{}_test.csv'.format(radius_find), index=False)


if __name__ == '__main__':
    curr_path = Path('')
    data_path = INPUT_PATH
    stac_path = INPUT_PATH + 'stac/'

    create_feature_tables_init()
    find_neigbours(10)
    class_distribution_for_neighbours(10)
    class_distribution_for_neighbours(100)
    class_distribution_in_radius_for_neighbours(1000)
    class_distribution_in_radius_for_neighbours(10000)


'''
area_of_building, map_id, pos_x, pos_y 
10 Neigbours: distance, type_of_neighbour, area, pos_x, pos_y  
'''