import geopandas as gpd
import pandas as pd


def combine_geojsons(data_dict):
    result = []
    for data in ['train_geojson', 'test_geojson']:
        geo = gpd.read_file(data_dict[list(data_dict.keys())[0]]['path'] + data_dict[list(data_dict.keys())[0]][data])
        geo['name'] = list(data_dict.keys())[0]
        geo.set_index('id', inplace=True)
        for name in list(data_dict.keys())[1:]:
            if 'test_geojson' not in data_dict[name]:
                continue
            geo_i = gpd.read_file(data_dict[name]['path'] + data_dict[name][data])
            geo_i['name'] = name
            geo_i.set_index('id', inplace=True)
            geo = pd.concat([geo, geo_i])
        result.append(geo)
    return result

def main():
    data_dir = 'data'
    data_dict = {'borde_rural':
                     {'path': f'{data_dir}/stac/colombia/borde_rural/',
                      'imagery': 'borde_rural_ortho-cog.tif',
                      'train_geojson': 'train-borde_rural.geojson',
                      'test_geojson': 'test-borde_rural.geojson'},
                 'borde_soacha':
                     {'path': f'{data_dir}/stac/colombia/borde_soacha/',
                      'imagery': 'borde_soacha_ortho-cog.tif',
                      'train_geojson': 'train-borde_soacha.geojson',
                      'test_geojson': 'test-borde_soacha.geojson'},
                 'mixco_1_and_ebenezer':
                     {'path': f'{data_dir}/stac/guatemala/mixco_1_and_ebenezer/',
                      'imagery': 'mixco_1_and_ebenezer_ortho-cog.tif',
                      'train_geojson': 'train-mixco_1_and_ebenezer.geojson',
                      'test_geojson': 'test-mixco_1_and_ebenezer.geojson'},
                 'mixco_3':
                     {'path': f'{data_dir}/stac/guatemala/mixco_3/',
                      'imagery': 'mixco_3_ortho-cog.tif',
                      'train_geojson': 'train-mixco_3.geojson',
                      'test_geojson': 'test-mixco_3.geojson'},
                 'castries':
                     {'path': f'{data_dir}/stac/st_lucia/castries/',
                      'imagery': 'castries_ortho-cog.tif',
                      'train_geojson': 'train-castries.geojson'},
                 'dennery':
                     {'path': f'{data_dir}/stac/st_lucia/dennery/',
                      'imagery': 'dennery_ortho-cog.tif',
                      'train_geojson': 'train-dennery.geojson',
                      'test_geojson': 'test-dennery.geojson'},
                 'gros_islet':
                     {'path': f'{data_dir}/stac/st_lucia/gros_islet/',
                      'imagery': 'gros_islet_ortho-cog.tif',
                      'train_geojson': 'train-gros_islet.geojson'}
                 }

    train_geojson, test_geojson = combine_geojsons(data_dict)
    print(f'Number of train samples - {len(train_geojson)}')
    print(f'Number of test samples - {len(test_geojson)}')

    train_geojson.reset_index(inplace=True)
    test_geojson.reset_index(inplace=True)
    train_geojson.to_file(f'{data_dir}/train.geojson', driver='GeoJSON')
    test_geojson.to_file(f'{data_dir}/test.geojson', driver='GeoJSON')


if __name__=="__main__":
    main()