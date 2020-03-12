import numpy as np
import pandas as pd


class DataFeed(object):
    """Class to feed images"""

    def __init__(self, df, data_load_func,
                 x_transform_func=None, x_preprocess_func=None,
                 y_transform_func=None, y_preprocess_func=None,
                 predict=False, multi_input=False, multi_output=False, scale_df=1, squeeze_single_elements=True):
        """Initialize class.

        Take a dataframe with images path in *path* column and
        annotations in *annot* column, pre-process images, and return
        a tuple of (images, targets) when the class is called.

        :param df: DataFrame.
            DataFrame with X,y data
        :param data_load_func: Function.
            Function to load data (X, y, info) from DataFrame row.
        :param x_transform_func: Function.
            Function or sequence(*Compose*) of functions to transform X.
        :param x_preprocess_func: Function.
            Function or sequence(*Compose*) of functions to pre-process X.
        :param y_transform_func: Function.
            Function or sequence(*Compose*) of functions to transform y.
        :param y_preprocess_func: Function.
            Function or sequence(*Compose*) of functions to pre-process y.
        :param predict: Boolean, default False.
            If True, only X is returned, else (X, y)
        :param multi_input: Boolean, default False.
            If True, NN input is [X1, ..., Xn]
        :param multi_output: Boolean, default False.
            If True, NN output is [y1, ..., ym]
        :param scale_df: Int, default 1
            Factor to increase the dataset scale_df times.
            Useful when samples for each df row are random.
        :param squeeze_single_elements: Boolean, default True.
            Whether to squeeze outputs when input as just 1 index instead
            of a list of indexes.
        """

        self.df = df.copy()
        assert scale_df >= 1 and type(scale_df) == int
        self.scale_df = scale_df
        if self.scale_df > 1:
            self.df = pd.concat([self.df, ] * self.scale_df, axis=0)
        self.df.reset_index(inplace=True, drop=True)

        self.data_load_func = data_load_func
        self.x_transform_func = x_transform_func if x_transform_func is not None else lambda x: x
        self.x_preprocess_func = x_preprocess_func if x_preprocess_func is not None else lambda x: x
        self.y_transform_func = y_transform_func if y_transform_func is not None else lambda x: x
        self.y_preprocess_func = y_preprocess_func if y_preprocess_func is not None else lambda x: x

        self.predict = predict
        self.multi_input = multi_input
        self.multi_output = multi_output
        self.squeeze_single_elements = squeeze_single_elements

        self.debug_mode = False

        # Samples weight
        if 'sample_weight' in self.df.columns:
            self.sample_weight = df.sample_weight.values
            print('--Using sample_weight')
        else:
            self.sample_weight = None

        # Dataset length
        self.length = len(self.df)

    def __getitem__(self, index, return_transformed=False):
        """Get item method"""

        # Make a list of indexes
        if not hasattr(index, '__iter__'):
            index = [index, ]

        # Get X, y, info
        x0_rsl, y0_rsl, info = zip(*[self.data_load_func(self.df.loc[s]) for s in index])
        if self.debug_mode:
            return [s for s in x0_rsl]  # convert tuple to list

        # Make X transformations
        x0_trs = None
        if self.multi_input:
            x0_rsl = zip(*x0_rsl)
            if isinstance(self.x_transform_func, list):  # Different transformation for each output
                x0_rsl = [[f(s2) for s2 in s1] for f, s1 in zip(self.x_transform_func, x0_rsl)]
            else:
                x0_rsl = [self.x_transform_func(s) for s in x0_rsl]
        else:
            x0_rsl = [self.x_transform_func(s) for s in x0_rsl]
        if return_transformed:
            x0_trs = [s for s in x0_rsl]

        # Pre-process X
        if self.multi_input:
            if isinstance(self.x_preprocess_func, list):  # Different transformation for each output
                x0_rsl = [f(s) for f, s in zip(self.x_preprocess_func, x0_rsl)]
            else:
                x0_rsl = [self.x_preprocess_func(s) for s in x0_rsl]
            # Squeeze tensor for length-1 lists
            if self.squeeze_single_elements and len(index) == 1:
                x0_rsl = [s[0] for s in x0_rsl]
        else:
            x0_rsl = self.x_preprocess_func(x0_rsl)
            # Squeeze tensor for length-1 lists
            if self.squeeze_single_elements and len(index) == 1:
                x0_rsl = x0_rsl[0]

        if self.predict:
            if return_transformed:
                return x0_rsl, 0, x0_trs
            return x0_rsl, 0

        # Make y transformations
        if self.multi_output:
            if len(index) > 1:
                y0_rsl = zip(*y0_rsl)
            else:
                y0_rsl = y0_rsl
            if isinstance(self.y_transform_func, list):  # Different transformation for each output
                y0_rsl = [np.squeeze(f(s), 0) for f, s in zip(self.y_transform_func, y0_rsl)]
            else:
                y0_rsl = [np.squeeze(self.y_transform_func(s), 0) for s in y0_rsl]
        else:
            y0_rsl = [np.squeeze(self.y_transform_func(s), 0) for s in y0_rsl]

        # Pre-process y0
        if self.multi_output:
            if isinstance(self.y_preprocess_func, list):  # Different pre-processing for each output
                y0_rsl = [f(s) for f, s in zip(self.y_preprocess_func, zip(*y0_rsl))]
            else:
                y0_rsl = [self.y_preprocess_func(s) for s in zip(*y0_rsl)]
            # Squeeze tensor for length-1 lists
            if self.squeeze_single_elements and len(index) == 1:
                y0_rsl = [s[0] for s in y0_rsl]
        else:
            y0_rsl = self.y_preprocess_func(y0_rsl)
            # Squeeze tensor for length-1 lists
            if self.squeeze_single_elements and len(index) == 1:
                y0_rsl = y0_rsl[0]

        # Sample weights
        if self.sample_weight is not None:
            w_rsl = self.sample_weight[index].astype(np.float32)
            if return_transformed:
                return x0_rsl, y0_rsl, w_rsl, x0_trs
            return x0_rsl, y0_rsl, w_rsl

        if return_transformed:
            return x0_rsl, y0_rsl, x0_trs
        return x0_rsl, y0_rsl

    def __len__(self):
        """Length method"""
        return self.length

    def debug_mode_on(self):
        """Activate debug mode"""
        self.debug_mode = True

    def debug_mode_off(self):
        """Desactivate debug mode"""
        self.debug_mode = False
