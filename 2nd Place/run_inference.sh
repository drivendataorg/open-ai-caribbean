# Install these packages if you don't have them
# apt-get install libgeos-dev
# apt-get install libvips libvips-dev

# Change this variable to location of your python (Anaconda)
export PATH="/var/anaconda3-temp/bin/"
# Change this vairable to location of your code
export PYTHONPATH="$PYTHONPATH:/var/test_caribean/"

# Main pipeline for inference
pip install -r requirements.txt
python3 data_preprocessing/r01_extract_image_data.py
python3 data_preprocessing/r02_find_neighbours.py
python3 cnn_v1_densenet121/r26_classification_d121_valid_kfold_224.py
python3 cnn_v2_irv2/r26_classification_irv2_valid_kfold_299.py
python3 cnn_v2_irv2/r26_classification_irv2_valid_kfold_299_v2.py
python3 cnn_v3_efficientnet_b4/r26_classification_efficientnet_valid.py
python3 cnn_v4_densenet169/r26_classification_densenet169_valid.py
python3 cnn_v5_resnet34/r26_classification_valid.py
python3 cnn_v6_seresnext50/r26_classification_valid.py
python3 cnn_v7_resnet50/r26_classification_valid.py
python3 gbm_classifiers/r16_run_xgboost.py
python3 gbm_classifiers/r17_run_lightgbm.py
python3 gbm_classifiers/r15_run_catboost.py
python3 r20_ensemble_avg.py