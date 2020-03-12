import os

import pandas as pd
import geopandas as gpd
from sklearn.metrics import log_loss

DATA_FOLDER = 'data'
PREDICTION_FOLDER = 'predictions'
FILE_TEMPLATE = 'L2A_{model}_{partition}.csv.gz'


def calculate_local_cv(predictions_df):
    predictions_df.sort_index(inplace=True)
    train_shapefile = gpd.read_file(f"{DATA_FOLDER}/train.geojson")
    train_shapefile.set_index('id', inplace=True)
    train_shapefile.sort_index(inplace=True)
    local_cv = log_loss(train_shapefile.roof_material.astype("category").cat.codes, predictions_df.values)
    print(f'Local CV: {local_cv:.4f}')


def blend_two_df(df1_file_name, df2_file_name, df1_coeff=0.5):
    df1 = pd.read_csv(os.path.join(PREDICTION_FOLDER, df1_file_name), index_col=0)
    df2 = pd.read_csv(os.path.join(PREDICTION_FOLDER, df2_file_name), index_col=0)
    return df1_coeff*df1 + (1-df1_coeff)*df2


def main():
    val_predictions = blend_two_df(FILE_TEMPLATE.format(model='cat009', partition='folds'),
                                   FILE_TEMPLATE.format(model='lgb009', partition='folds'))
    test_predictions = blend_two_df(FILE_TEMPLATE.format(model='cat009', partition='test'),
                                    FILE_TEMPLATE.format(model='lgb009', partition='test'))
    calculate_local_cv(val_predictions)
    final_submission_file_name = os.path.join(PREDICTION_FOLDER, 'submission_cat_lgb_009.csv.gz')
    test_predictions.to_csv(final_submission_file_name, index=True)
    print(f'Final predictions saved in file: {final_submission_file_name}')


if __name__ == '__main__':
    main()
