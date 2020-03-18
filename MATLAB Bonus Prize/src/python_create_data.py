import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
import pandas as pd
import cv2
import os


def extract_train_roofs(fpath_tiff, fpath_geojson, save_img_path, save_mask_path):
    df_roof_geometries = gpd.read_file(fpath_geojson)

    with rasterio.open(fpath_tiff) as tiff:
        tiff_crs = tiff.crs.data
        df_roof_geometries['projected_geometry'] = (df_roof_geometries['geometry'].to_crs(tiff_crs))

    roof_geometries = (df_roof_geometries[['id', 'projected_geometry', 'roof_material']].values)

    with rasterio.open(fpath_tiff) as tiff:
        for roof_id, projected_geometry, roof_material in roof_geometries:
            roof_image, _ = mask(tiff, [projected_geometry], crop=True, pad=True, filled=False, pad_width=0)
            roof_image = np.transpose(roof_image, (1, 2, 0))
            roof_mask, _ = mask(tiff, [projected_geometry], crop=True, pad=True, filled=True, pad_width=0)
            roof_mask = np.transpose(roof_mask, (1, 2, 0))
            roof_mask = roof_mask[..., 0] > 0
            
            save_dir =  save_img_path + roof_material
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            image_name = save_dir + '/' + roof_id + '.png'
            mask_name = save_mask_path + roof_id + '_mask.png'

            roof_rgb = roof_image[:, :, 0:3]
            roof_bgr = cv2.cvtColor(roof_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(image_name, roof_bgr)
            cv2.imwrite(mask_name, 255*roof_mask)


def extract_test_roofs(fpath_tiff, fpath_geojson, save_img_path, save_mask_path):
    df_roof_geometries = gpd.read_file(fpath_geojson)

    with rasterio.open(fpath_tiff) as tiff:
        tiff_crs = tiff.crs.data
        df_roof_geometries['projected_geometry'] = (df_roof_geometries['geometry'].to_crs(tiff_crs))

    roof_geometries = (df_roof_geometries[['id', 'projected_geometry']].values)

    with rasterio.open(fpath_tiff) as tiff:
        for roof_id, projected_geometry in roof_geometries:
            roof_image, _ = mask(tiff, [projected_geometry], crop=True, pad=True, filled=False, pad_width=0)
            roof_image = np.transpose(roof_image, (1, 2, 0))
            roof_mask, _ = mask(tiff, [projected_geometry], crop=True, pad=True, filled=True, pad_width=0)
            roof_mask = np.transpose(roof_mask, (1, 2, 0))
            roof_mask = roof_mask[..., 0] > 0
            

            image_name = save_img_path + roof_id + '.png'
            mask_name = save_mask_path + roof_id + '_mask.png'

            roof_rgb = roof_image[:, :, 0:3]
            roof_bgr = cv2.cvtColor(roof_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(image_name, roof_bgr)
            cv2.imwrite(mask_name, 255*roof_mask)


#Create Training data
save_img_path = '../data/processed/train/'
if not os.path.exists(save_img_path):
                os.makedirs(save_img_path)
save_mask_path = '../data/processed/data_mask/'
if not os.path.exists(save_mask_path):
                os.makedirs(save_mask_path)
        

fpath_tiff = '../data/raw/stac/colombia/borde_rural/borde_rural_ortho-cog.tif' # put filepath of tiff here
fpath_geojson = '../data/raw/stac/colombia/borde_rural/train-borde_rural.geojson' # put filepath of geojson here
extract_train_roofs(fpath_tiff, fpath_geojson, save_img_path, save_mask_path)

fpath_tiff = '../data/raw/stac/colombia/borde_soacha/borde_soacha_ortho-cog.tif' # put filepath of tiff here
fpath_geojson = '../data/raw/stac/colombia/borde_soacha/train-borde_soacha.geojson' # put filepath of geojson here
extract_train_roofs(fpath_tiff, fpath_geojson, save_img_path, save_mask_path)

fpath_tiff = '../data/raw/stac/guatemala/mixco_1_and_ebenezer/mixco_1_and_ebenezer_ortho-cog.tif' # put filepath of tiff here
fpath_geojson = '../data/raw/stac/guatemala/mixco_1_and_ebenezer/train-mixco_1_and_ebenezer.geojson' # put filepath of geojson here
extract_train_roofs(fpath_tiff, fpath_geojson, save_img_path, save_mask_path)

fpath_tiff = '../data/raw/stac/guatemala/mixco_3/mixco_3_ortho-cog.tif' # put filepath of tiff here
fpath_geojson = '../data/raw/stac/guatemala/mixco_3/train-mixco_3.geojson' # put filepath of geojson here
extract_train_roofs(fpath_tiff, fpath_geojson, save_img_path, save_mask_path)

fpath_tiff = '../data/raw/stac/st_lucia/dennery/dennery_ortho-cog.tif' # put filepath of tiff here
fpath_geojson = '../data/raw/stac/st_lucia/dennery/train-dennery.geojson' # put filepath of geojson here
extract_train_roofs(fpath_tiff, fpath_geojson, save_img_path, save_mask_path)


#Create Testing data
save_img_path = '../data/processed/test/'
if not os.path.exists(save_img_path):
                os.makedirs(save_img_path)
save_mask_path = '../data/processed/roof_mask/'
if not os.path.exists(save_mask_path):
                os.makedirs(save_mask_path)
        

fpath_tiff = '../data/raw/stac/colombia/borde_rural/borde_rural_ortho-cog.tif' # put filepath of tiff here
fpath_geojson = '../data/raw/stac/colombia/borde_rural/test-borde_rural.geojson' # put filepath of geojson here
extract_test_roofs(fpath_tiff, fpath_geojson, save_img_path, save_mask_path)

fpath_tiff = '../data/raw/stac/colombia/borde_soacha/borde_soacha_ortho-cog.tif' # put filepath of tiff here
fpath_geojson = '../data/raw/stac/colombia/borde_soacha/test-borde_soacha.geojson' # put filepath of geojson here
extract_test_roofs(fpath_tiff, fpath_geojson, save_img_path, save_mask_path)

fpath_tiff = '../data/raw/stac/guatemala/mixco_1_and_ebenezer/mixco_1_and_ebenezer_ortho-cog.tif' # put filepath of tiff here
fpath_geojson = '../data/raw/stac/guatemala/mixco_1_and_ebenezer/test-mixco_1_and_ebenezer.geojson' # put filepath of geojson here
extract_test_roofs(fpath_tiff, fpath_geojson, save_img_path, save_mask_path)

fpath_tiff = '../data/raw/stac/guatemala/mixco_3/mixco_3_ortho-cog.tif' # put filepath of tiff here
fpath_geojson = '../data/raw/stac/guatemala/mixco_3/test-mixco_3.geojson' # put filepath of geojson here
extract_test_roofs(fpath_tiff, fpath_geojson, save_img_path, save_mask_path)

fpath_tiff = '../data/raw/stac/st_lucia/dennery/dennery_ortho-cog.tif' # put filepath of tiff here
fpath_geojson = '../data/raw/stac/st_lucia/dennery/test-dennery.geojson' # put filepath of geojson here
extract_test_roofs(fpath_tiff, fpath_geojson, save_img_path, save_mask_path)