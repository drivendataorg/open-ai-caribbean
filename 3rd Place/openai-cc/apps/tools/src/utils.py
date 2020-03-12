import json

import rasterio
from pyproj import Proj


def found_epsg(tiff_path):
    with rasterio.open(tiff_path) as tiff:
        return tiff.crs.to_epsg()


def transform_coordinates(path_to_geojson, epsg):
    def _transform_one_json_coordinates(coordinates_lists, transform):
        transformed_coordinates_lists = []
        for coordinates_list in coordinates_lists:
            transformed_coordinates_list = []
            for coordinate in coordinates_list:
                coordinate = tuple(coordinate)
                transformed_coordinate = list(transform(coordinate[0], coordinate[1]))
                transformed_coordinates_list.append(transformed_coordinate)
            transformed_coordinates_lists.append(transformed_coordinates_list)

        return transformed_coordinates_lists

    def _filter_redundant(geojson):
        return {
            'id': geojson.get('id'),
            'roof_material': geojson['properties'].get('roof_material'),
            'verified': geojson['properties'].get('verified'),
            'coordinates': geojson['geometry'].get('coordinates'),
        }

    proj = Proj(init=f'epsg:{epsg}')
    with open(path_to_geojson) as fp:
        polygons = json.load(fp)

    features_processed = []
    for features in polygons['features']:
        features = _filter_redundant(features)
        features['coordinates'] = _transform_one_json_coordinates(features['coordinates'], proj)
        features_processed.append(features)

    return features_processed
