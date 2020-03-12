import pandas as pd
import numpy as np

import geopandas as gpd
from shapely.geometry import Polygon


def get_data_dict(data_dir):
    """Return dictionary with information about dataset

    :param data_dir: directory to project data
    """
    data_dict = {'borde_rural':
                     {'path': f'{data_dir}/stac/colombia/borde_rural/',
                      'imagery': 'borde_rural_ortho-cog.tif',
                      'train_geojson': 'train-borde_rural.geojson',
                      'test_geojson': 'test-borde_rural.geojson',
                      'country': 'colombia'},
                 'borde_soacha':
                     {'path': f'{data_dir}/stac/colombia/borde_soacha/',
                      'imagery': 'borde_soacha_ortho-cog.tif',
                      'train_geojson': 'train-borde_soacha.geojson',
                      'test_geojson': 'test-borde_soacha.geojson',
                      'country': 'colombia'},
                 'mixco_1_and_ebenezer':
                     {'path': f'{data_dir}/stac/guatemala/mixco_1_and_ebenezer/',
                      'imagery': 'mixco_1_and_ebenezer_ortho-cog.tif',
                      'train_geojson': 'train-mixco_1_and_ebenezer.geojson',
                      'test_geojson': 'test-mixco_1_and_ebenezer.geojson',
                      'country': 'guatemala'},
                 'mixco_3':
                     {'path': f'{data_dir}/stac/guatemala/mixco_3/',
                      'imagery': 'mixco_3_ortho-cog.tif',
                      'train_geojson': 'train-mixco_3.geojson',
                      'test_geojson': 'test-mixco_3.geojson',
                      'country': 'guatemala'},
                 'castries':
                     {'path': f'{data_dir}/stac/st_lucia/castries/',
                      'imagery': 'castries_ortho-cog.tif',
                      'train_geojson': 'train-castries.geojson',
                      'country': 'st_lucia'},
                 'dennery':
                     {'path': f'{data_dir}/stac/st_lucia/dennery/',
                      'imagery': 'dennery_ortho-cog.tif',
                      'train_geojson': 'train-dennery.geojson',
                      'test_geojson': 'test-dennery.geojson',
                      'country': 'st_lucia'},
                 'gros_islet':
                     {'path': f'{data_dir}/stac/st_lucia/gros_islet/',
                      'imagery': 'gros_islet_ortho-cog.tif',
                      'train_geojson': 'train-gros_islet.geojson',
                      'country': 'st_lucia'}
                 }
    return data_dict


def combine_geojsons(data_dict):
    """Combine train and test geojson data

    :param data_dict: directory to project data
    """
    result = []
    for data in ['train_geojson', 'test_geojson']:
        geos = []
        for loc in list(data_dict.keys()):
            # print(loc)
            if data not in data_dict[loc]:
                continue
            geo = gpd.read_file(data_dict[loc]['path'] + data_dict[loc][data])
            geo['name'] = loc
            geo['country'] = data_dict[loc]['country']
            geos.append(geo)
            # print(data_dict[loc]['path'], data_dict[loc][data], len(geos))
        geos = pd.concat(geos, axis=0)
        geos.set_index('id', inplace=True)
        result.append(geos)
    return result


def get_dataset_df(data_dir, verbose=True):
    """Return DataFrame with dataset

    :param data_dir: directory to project data
    :param verbose: whether to print out info
    """
    # Get data_dict
    data_dict = get_data_dict(data_dir)

    # Get train/test sets
    train_geojson, test_geojson = combine_geojsons(data_dict)
    if verbose:
        print(f'Number of train samples - {len(train_geojson)}')
        print(f'Number of test samples - {len(test_geojson)}')

    # Add train/test columns
    train_geojson['train'], train_geojson['test'] = True, False
    test_geojson['train'], test_geojson['test'] = False, True
    test_geojson['verified'] = None

    # Get class_ids
    class_names = train_geojson.roof_material.unique().tolist()
    class_names.sort()
    class_ids = range(len(class_names))
    class_to_id = {s1: s2 for s1, s2 in zip(class_names, class_ids)}
    train_geojson['class_id'] = train_geojson.roof_material.map(lambda x: class_to_id[x])
    test_geojson['class_id'] = None

    # Concatenate train/test datasets
    dset_geojson = pd.concat([train_geojson, test_geojson], axis=0, sort=True)
    dset_geojson.reset_index(inplace=True)

    # Get location_ids
    loc_names = dset_geojson.name.unique().tolist()
    loc_names.sort()
    loc_ids = range(len(loc_names))
    name_to_id = {s1: s2 for s1, s2 in zip(loc_names, loc_ids)}
    dset_geojson['loc_id'] = dset_geojson.name.map(lambda x: name_to_id[x])

    return dset_geojson


def polygon_to_tuples(polygon):
    """Transform polygon to list of tuples (x, y)

    :param polygon: shapely Polygon
    """
    x, y = polygon.convex_hull.exterior.coords.xy
    x = list(x)
    y = list(y)
    return list(zip(x, y))


def polygon_to_lists(polygon):
    """Transform polygon to tuple of list (x, y)

    :param polygon: shapely Polygon
    """
    x, y = polygon.convex_hull.exterior.coords.xy
    x = list(x)
    y = list(y)
    return x, y


def transform_coordinates(coordinates_lists, transform):
    """Transform the coordinates with the provided `affine_transform`

    :param coordinates_lists: list of lists of coordinates
    :param transform: transformation to apply
    """

    transformed_coordinates_lists = []
    for coordinates_list in coordinates_lists:
        transformed_coordinates_list = []
        coordinates_list = polygon_to_tuples(coordinates_list)
        for coordinate in coordinates_list:
            coordinate = tuple(coordinate)
            transformed_coordinate = list(transform(coordinate[0], coordinate[1]))
            transformed_coordinates_list.append(transformed_coordinate)
        poly = Polygon(transformed_coordinates_list)
        transformed_coordinates_lists.append(poly)
    return transformed_coordinates_lists


def xy2pix(transform, xs, ys):
    """Transform the x, y coordinates pixel position col, row

    :param xs: list of x of coordinates
    :param ys: list of y of coordinates
    :param transform: transformation to apply
    """
    det = transform.a * transform.e - transform.b * transform.d
    cols = []
    rows = []
    for x, y in zip(xs, ys):
        col = (x * transform.e - y * transform.b +
               transform.b * transform.yoff - transform.xoff * transform.e) / det
        row = (-x * transform.d + y * transform.a +
               -transform.a * transform.yoff + transform.d * transform.xoff) / det
        cols.append(int(np.round(col, 0)))
        rows.append(int(np.round(row, 0)))
    return cols, rows
