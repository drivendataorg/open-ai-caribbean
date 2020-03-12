import os

import numpy as np

import rasterio
import rasterio.mask as mask
from pyproj import Proj
from PIL import Image
from PIL import ImageDraw

from torch.utils.data import Dataset

from src.data.geodata_utils import transform_coordinates, xy2pix, polygon_to_lists


class RoofDataset(Dataset):

    def __init__(self, dataframe, datadict, mask_image=True, buffer=0, plot_polygons=False,
                 add_location=False, add_extra_features=False,
                 annot_column='class_id', return_ohe=False, nb_classes=None):
        """Initialize class.

        Take a dataframe with images path in *path*

        :param dataframe: DataFrame.
            DataFrame with dataset
        :param datadict: Dictionary.
            Dictionary with dataset info
        :param mask_image: Boolean, default True.
            Whether to mask the part of exterior of the polygon.
        :param buffer: Float, default 0.
            Distance to increase/reduce the polygon extension.
        :param plot_polygons: Boolean, default False.
            Whether to plot the actual polygon in the output image.
        :param add_location: Boolean, default False.
            Whether to add the location to the outputs.
        :param add_extra_features: Boolean, default False.
            Whether to add extra features to the outputs.
        :param annot_column: String, default 'class_id'.
            DataFrame column's name with the annotations.
        :param return_ohe: Boolean, default False.
            Whether to return the annotations as One Hot Encoded.
        :param nb_classes: Boolean, default False.
            Total number of classes.
        """

        self.df = dataframe.copy()
        self.datadict = datadict.copy()
        self.mask_image = mask_image
        self.buffer = buffer
        self.plot_polygons = plot_polygons
        self.add_location = add_location
        self.add_extra_features = add_extra_features
        self.annot_column = annot_column
        if return_ohe:
            assert nb_classes is not None
        self.return_ohe = return_ohe
        self.nb_classes = nb_classes

    def __len__(self):
        return len(self.df)

    def __getitem__(self, ix):
        irow = self.df.iloc[ix]
        return self.read(irow)

    def read(self, irow):
        loc_name = irow['name']
        loc_tiff_path = os.path.join(self.datadict[loc_name]['path'], self.datadict[loc_name]['imagery'])

        with rasterio.open(loc_tiff_path) as src:
            proj = Proj(init=f'epsg:{src.crs.to_epsg()}')
            geometry = transform_coordinates([irow['geometry'], ], proj)
            buffer_geometry = [s1.buffer(self.buffer) for s1 in geometry]

            if self.mask_image:
                out_image, out_transform = rasterio.mask.mask(src, buffer_geometry, crop=True, all_touched=False)
            else:
                patch_mask, out_transform, window = rasterio.mask.raster_geometry_mask(src, buffer_geometry, crop=True,
                                                                                       all_touched=False)
                out_image = src.read([1, 2, 3], window=window)
            im = Image.fromarray(np.rollaxis(out_image, 0, 3)).convert('RGB')

        if self.plot_polygons:
            # Plot polygons
            img_geometry = [xy2pix(out_transform, *polygon_to_lists(s1)) for s1 in geometry]
            img_geometry = [[(s1, s2) for s1, s2 in zip(*s0)] for s0 in img_geometry]
            pdraw = ImageDraw.Draw(im)
            [pdraw.polygon(s1, outline=(255, 0, 0)) for s1 in img_geometry]

        if self.return_ohe:
            annot = np.zeros((1, self.nb_classes), dtype=np.float32)
            annot[0, irow[self.annot_column]] = 1
        else:
            annot = irow[self.annot_column]

        # return PIL_image, label, info dictionary
        if self.add_extra_features:
            extra_input = irow[[s for s in irow.keys() if "feat_" in s]].values.astype(np.float32)
            return (im, extra_input), annot, {}
        if self.add_location:
            return (im, irow.loc_id), annot, {}
        return im, annot, {}

    def read_id(self, image_id):
        irow = self.df[self.df.id == image_id].iloc[0]
        return self.read(irow)

    def read_random(self):
        irow = self.df.iloc[np.random.choice(range(len(self.df)))]
        return self.read(irow)


def _azimuth(point1, point2):
    angle = np.arctan2(point2[0] - point1[0], point2[1] - point1[1])
    return np.degrees(angle) if angle > 0 else np.degrees(angle) + 360


def _lenght(point1, point2):
    return np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)


def azimut(points):
    s1, s2 = zip(*[(_lenght(points[i1], points[i1 + 1]), _azimuth(points[i1], points[i1 + 1])) for i1 in range(2)])
    az = s2[np.argmax(s1)]
    az = az - 180 if az >= 180 else az
    return az


def aspect_ratio(points):
    s1, s2 = zip(*[(_lenght(points[i1], points[i1 + 1]), _azimuth(points[i1], points[i1 + 1])) for i1 in range(2)])
    return max(s1) / min(s1)


def relative_area(pol):
    envelope = list(pol.envelope.boundary.coords)
    return pol.area / max([abs(_lenght(envelope[0], envelope[1])), abs(_lenght(envelope[1], envelope[2]))])
