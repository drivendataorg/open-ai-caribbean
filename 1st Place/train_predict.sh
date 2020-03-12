#!/usr/bin/env bash
set -e

echo "################ 1. Create environment and installing dependencies ####################"
pip3 install pipenv
pipenv install --skip-lock
echo "#######################################################################################"
echo

echo "################ 2. Combine train/test geojson from several locations #################"
pipenv run python src/data/preprocess_input_data.py
echo "#######################################################################################"
echo

echo "################ 3. Train/predict first level CNN models ##############################"
pipenv run python src/train_L1A_models.py --model_id A06 --folds A
pipenv run python src/train_L1A_models.py --model_id A06 --folds B
pipenv run python src/train_L1A_models.py --model_id A06 --folds C
pipenv run python src/train_L1A_models.py --model_id A06 --folds D
pipenv run python src/train_L1A_models.py --model_id A06 --folds E
pipenv run python src/train_L1A_models.py --model_id A06 --folds F
pipenv run python src/train_L1A_models.py --model_id A06 --folds G
pipenv run python src/train_L1A_models.py --model_id A06 --folds H

pipenv run python src/train_L1A_models.py --model_id A10 --folds A
pipenv run python src/train_L1A_models.py --model_id A10 --folds B
pipenv run python src/train_L1A_models.py --model_id A10 --folds C
pipenv run python src/train_L1A_models.py --model_id A10 --folds D
pipenv run python src/train_L1A_models.py --model_id A10 --folds E
pipenv run python src/train_L1A_models.py --model_id A10 --folds F
pipenv run python src/train_L1A_models.py --model_id A10 --folds G
pipenv run python src/train_L1A_models.py --model_id A10 --folds H

pipenv run python src/train_L1A_models.py --model_id A11a --folds A
pipenv run python src/train_L1A_models.py --model_id A11a --folds B
pipenv run python src/train_L1A_models.py --model_id A11a --folds C
pipenv run python src/train_L1A_models.py --model_id A11a --folds D
pipenv run python src/train_L1A_models.py --model_id A11a --folds E
pipenv run python src/train_L1A_models.py --model_id A11a --folds F
pipenv run python src/train_L1A_models.py --model_id A11a --folds G
pipenv run python src/train_L1A_models.py --model_id A11a --folds H

pipenv run python src/train_L1A_models.py --model_id C06g --folds A
pipenv run python src/train_L1A_models.py --model_id C06g --folds B
pipenv run python src/train_L1A_models.py --model_id C06g --folds C
pipenv run python src/train_L1A_models.py --model_id C06g --folds D
pipenv run python src/train_L1A_models.py --model_id C06g --folds E
pipenv run python src/train_L1A_models.py --model_id C06g --folds F
pipenv run python src/train_L1A_models.py --model_id C06g --folds G
pipenv run python src/train_L1A_models.py --model_id C06g --folds H

echo "################ 3.1 Concatenate L1 predictions #######################################"
pipenv run python src/concatenate_L1A_predictions.py --model_id A06
pipenv run python src/concatenate_L1A_predictions.py --model_id A10
pipenv run python src/concatenate_L1A_predictions.py --model_id A11a
pipenv run python src/concatenate_L1A_predictions.py --model_id C06g
echo "#######################################################################################"
echo

echo "################ 4. Train/predict second level models #################################"
pipenv run python src/2_catboost_dfg_009.py
pipenv run python src/2_lightgbm_dfg_009.py
echo "#######################################################################################"
echo

echo "################ 5. Blend second level models #########################################"
pipenv run python src/final_ensemble.py
echo "#######################################################################################"