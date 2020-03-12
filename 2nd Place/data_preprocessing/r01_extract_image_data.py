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


TRAIN_PATH = OUTPUT_PATH + 'train_img/'
if not os.path.isdir(TRAIN_PATH):
    os.mkdir(TRAIN_PATH)
TRAIN_PATH_ALPHA = TRAIN_PATH + 'alpha/'
if not os.path.isdir(TRAIN_PATH_ALPHA):
    os.mkdir(TRAIN_PATH_ALPHA)
TEST_PATH = OUTPUT_PATH + 'test_img/'
if not os.path.isdir(TEST_PATH):
    os.mkdir(TEST_PATH)
TEST_PATH_ALPHA = TEST_PATH + 'alpha/'
if not os.path.isdir(TEST_PATH_ALPHA):
    os.mkdir(TEST_PATH_ALPHA)


total_multi_train = 0
total_multi_test = 0


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

    for train_path, tiff_path in zip(train_geo_paths, tiff_paths):
        print('Go for: {}'.format(train_path))
        print('Go for: {}'.format(tiff_path))
        if str(train_path) == 'nan':
            continue
        train_path = str(data_path + '/' + train_path)
        tiff_path = str(data_path + '/' + tiff_path)
        alias = train_path.split('/')[-2]
        df = gpd.read_file(train_path)
        df['train_path'] = train_path
        df['tiff_path'] = tiff_path
        df['alias'] = alias
        with rasterio.open(tiff_path) as tiff:
            epsg = tiff.crs.to_epsg()
            df['epsg'] = epsg
        dfs.append(df)

    df = pd.concat(dfs)
    print(df)

    return df


def process_single_row(train_df, index, type):
    global total_multi_train, total_multi_test

    delta = 100
    sample = train_df.iloc[index]

    # print('Go for: {}'.format(sample.id))
    if type == 'train':
        out_png = TRAIN_PATH + '{}.png'.format(sample.id)
    else:
        out_png = TEST_PATH + '{}.png'.format(sample.id)

    polygon = sample.geometry
    if 'MULTI' in str(polygon):
        if type == 'train':
            total_multi_train += 1
        else:
            total_multi_test += 1
        print('Some multi here', sample.id)
    else:
        if os.path.isfile(out_png):
            return

    tuples = polygon_to_tuples(polygon)
    proj = Proj(init='epsg:{}'.format(sample.epsg))
    tiff_path = sample.tiff_path
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
    if 0:
        if xmin < delta:
            print('Too small xmin')
        if ymin < delta:
            print('Too small ymin')
        if xmax > tiff.width - delta:
            print('Too big xmax')
        if ymax > tiff.height - delta:
            print('Too big ymax')

    window = ((int(xmin), int(xmax)), (int(ymin), int(ymax)))
    xmin_ex = xmin - delta
    xmax_ex = xmax + delta
    ymin_ex = ymin - delta
    ymax_ex = ymax + delta
    window_expanded = ((int(xmin_ex), int(xmax_ex)), (int(ymin_ex), int(ymax_ex)))

    # print(pixels)
    # print(window)
    b = tiff.read(1, window=window_expanded)
    g = tiff.read(2, window=window_expanded)
    r = tiff.read(3, window=window_expanded)
    alpha = tiff.read(4, window=window_expanded)
    # print(r.shape, r.min(), r.max(), r.mean())
    # print(g.shape, g.min(), g.max(), g.mean())
    # print(b.shape, b.min(), b.max(), b.mean())
    # print(alpha.shape, alpha.min(), alpha.max(), alpha.mean())
    if alpha.max() > 255:
        print('Big alpha!')
        exit()

    mask = np.zeros(b.shape, dtype=np.int32)
    poly = np.array(pixels)
    poly[:, 0] -= xmin_ex
    poly[:, 1] -= ymin_ex
    if 1:
        r1, c1 = poly[:, 0].copy(), poly[:, 1].copy()
        poly[:, 0], poly[:, 1] = c1, r1
    # print(poly)

    mask = cv2.fillConvexPoly(mask, poly, 255)
    img = np.stack((b, g, r), axis=2)
    # show_image(img)
    # show_image(mask)
    if type == 'train':
        cv2.imwrite(out_png, img)
        cv2.imwrite(TRAIN_PATH + '{}_mask.png'.format(sample.id), mask.astype(np.uint8))
        cv2.imwrite(TRAIN_PATH_ALPHA + '{}.png'.format(sample.id), alpha.astype(np.uint8))
    else:
        cv2.imwrite(out_png, img)
        cv2.imwrite(TEST_PATH + '{}_mask.png'.format(sample.id), mask.astype(np.uint8))
        cv2.imwrite(TEST_PATH_ALPHA + '{}.png'.format(sample.id), alpha.astype(np.uint8))


if __name__ == '__main__':
    curr_path = Path('')
    data_path = INPUT_PATH
    stac_path = INPUT_PATH + 'stac/'

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

    if 1:
        train_df = get_train_test_df(train_geo_paths,
                                tiff_paths,
                                data_path)

        print(list(train_df.columns))
        train_df['geometry'] = train_df['geometry'].astype(object)
        train_df.to_csv(OUTPUT_PATH + 'train.csv', index=False)

        for i in range(len(train_df)):
            process_single_row(train_df, i, 'train')

        print('Missed train because of MULTI: {}'.format(total_multi_train))


    if 1:
        test_df = get_train_test_df(test_geo_paths,
                                tiff_paths,
                                data_path)

        test_df['geometry'] = test_df['geometry'].astype(object)
        test_df.to_csv(OUTPUT_PATH + 'test.csv', index=False)

        for i in range(len(test_df)):
            process_single_row(test_df, i, 'test')

        print('Missed because of MULTI: {}'.format(total_multi_test))
        test_df.to_csv(OUTPUT_PATH + 'test.csv', index=False)