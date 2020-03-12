* Software Requirements

Main requirements: Python 3.5+, keras 2.2+, Tensorflow 1.13+, classification_models (latest from git), efficientnet (latest from git)
Other requirements: numpy, pandas, opencv-python, scipy, sklearn, pyvips, pyproj, geopandas, pathlib, shapely
You need to have CUDA 10.0 installed
Solution was tested on Anaconda3-2019.10-Linux-x86_64.sh: https://www.anaconda.com/distribution/

* Hardware requirements

* All batch sizes for Neural nets are tuned to be used on NVIDIA GTX 1080 Ti 11 GB card. To use code with other GPUs with less memory - decrease batch size accordingly.
* At some point during image cache generation code temporary could require around 128 GB of RAM memory (probably swap could handle this).

* How to run:

Code expects all input files in "input/" directory. Fix paths in a00_common_functions.py if needed.
All r*.py files must be run one by one. All intermediate folders will be created automatically.

Full pipeline:
python data_preprocessing/r01_extract_image_data.py
python data_preprocessing/r02_find_neighbours.py
python cnn_v1_densenet121/r16_classification_d121_train_kfold_224.py
python cnn_v1_densenet121/r26_classification_d121_valid_kfold_224.py
python cnn_v2_irv2/r16_classification_irv2_train_kfold_299.py
python cnn_v2_irv2/r17_classification_irv2_train_kfold_299_v2.py
python cnn_v2_irv2/r26_classification_irv2_valid_kfold_299.py
python cnn_v2_irv2/r26_classification_irv2_valid_kfold_299_v2.py
python cnn_v3_efficientnet_b4/r17_classification_efficientnet_train_kfold_380.py
python cnn_v3_efficientnet_b4/r26_classification_efficientnet_valid.py
python cnn_v4_densenet169/r17_classification_densenet169_train_kfold_224.py
python cnn_v4_densenet169/r26_classification_densenet169_valid.py
python cnn_v5_resnet34/r17_classification_train_kfold_224.py
python cnn_v5_resnet34/r26_classification_valid.py
python cnn_v6_seresnext50/r17_classification_train_kfold_224.py
python cnn_v6_seresnext50/r26_classification_valid.py
python cnn_v7_resnet50/r17_classification_train_kfold_224.py
python cnn_v7_resnet50/r26_classification_valid.py
python gbm_classifiers/r15_run_catboost.py
python gbm_classifiers/r16_run_xgboost.py
python gbm_classifiers/r17_run_lightgbm.py
python r20_ensemble_avg.py

Only inference part (starting from neural net models):
python data_preprocessing/r01_extract_image_data.py
python data_preprocessing/r02_find_neighbours.py
python cnn_v1_densenet121/r26_classification_d121_valid_kfold_224.py
python cnn_v2_irv2/r26_classification_irv2_valid_kfold_299.py
python cnn_v2_irv2/r26_classification_irv2_valid_kfold_299_v2.py
python cnn_v3_efficientnet_b4/r26_classification_efficientnet_valid.py
python cnn_v4_densenet169/r26_classification_densenet169_valid.py
python cnn_v5_resnet34/r26_classification_valid.py
python cnn_v6_seresnext50/r26_classification_valid.py
python cnn_v7_resnet50/r26_classification_valid.py
python gbm_classifiers/r15_run_catboost.py
python gbm_classifiers/r16_run_xgboost.py
python gbm_classifiers/r17_run_lightgbm.py
python r20_ensemble_avg.py

There is file run_inference.sh - which do all the stuff including pip installation of required modules etc.

Change this variable to location of your python (Anaconda)
export PATH="/var/anaconda3-temp/bin/"
Change this vairable to location of your code
export PYTHONPATH="$PYTHONPATH:/var/test_caribean/"

* Notes about a code

1) Change "ONLY_INFERENCE" constant in a00_common_functions.py to True for inference without training. You need to use the same
KFold splits, which will be read from cache. You need to use my KFold splits in modified_data folder - it needed to generate
the same feature files as mine for 2nd level models. Because for any other split as it was for training validation files which used in 2nd level model
as input data will be invalid.
2) Python files within different cnn_* folders can be run in parallel. E.g. you can train different neural networks independently.
3) It's only 3 networks enough for good accuracy: cnn_v1_densenet121, cnn_v2_irv2, cnn_v3_efficientnet_b4. All others improve overall result at local validation but lead to the same score on LB.

* File sizes and content
After running:
python data_preprocessing/r01_extract_image_data.py
python data_preprocessing/r02_find_neighbours.py

You must have the following:
Folder "modified_data/train_img/" - must contain 45106 images
Folder "modified_data/test_img/" - must contain 14650 images
File "modified_data/test.csv" - 3353297 bytes
File "modified_data/train.csv" - 11564407 bytes

Folder features:
   240068 Feb 20 09:36 neighbours_clss_distribution_100_test.csv
   727788 Feb 20 09:35 neighbours_clss_distribution_100_train.csv
   212523 Feb 20 09:30 neighbours_clss_distribution_10_test.csv
   654106 Feb 20 09:30 neighbours_clss_distribution_10_train.csv
   789213 Feb 20 10:50 neighbours_clss_distribution_radius_10000_test.csv
  2210199 Feb 20 10:28 neighbours_clss_distribution_radius_10000_train.csv
   542851 Feb 20 09:37 neighbours_clss_distribution_radius_1000_test.csv
  1379139 Feb 20 09:37 neighbours_clss_distribution_radius_1000_train.csv
  4179947 Feb 20 09:30 neighbours_feat_10_test.csv
 12880193 Feb 20 09:30 neighbours_feat_10_train.csv
   402323 Feb 20 09:29 test_neigbours.csv
  1685752 Feb 20 09:29 train_neigbours.csv

After runnung
python cnn_v1_densenet121/r26_classification_d121_valid_kfold_224.py

Folder "features":
  1628828 Feb 20 19:31 d121_kfold_valid_TTA_32_5.csv
  1628828 Feb 20 19:31 d121_kfold_valid_TTA_32_5_loss_0.430336_acc_0.837122.csv
   507955 Feb 20 20:57 d121_v2_test_TTA_32_5.csv
