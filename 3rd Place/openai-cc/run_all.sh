#!/usr/bin/env bash

#OPENAI_CONFIG=./configs/config-dense121.yml ./openai-cli run model.train
#OPENAI_CONFIG=./configs/config-rn34.yml ./openai-cli run model.train
#OPENAI_CONFIG=./configs/config-inceptionv3.yml ./openai-cli run model.train
#OPENAI_CONFIG=./configs/config-dense161-dropout.yml ./openai-cli run model.train
#OPENAI_CONFIG=./configs/config-b0.yml ./openai-cli run model.train

#OPENAI_CONFIG=./configs/config-b4.yml ./openai-cli run model.train
#OPENAI_CONFIG=./configs/config-se_rnxt50.yml ./openai-cli run model.train
#CUDA_VISIBLE_DEVICES=1 OPENAI_CONFIG=./configs/config-inceptionrnv2.yml ./openai-cli run model.train

OPENAI_CONFIG=./configs/config-dense121.yml ./openai-cli run model.inference
OPENAI_CONFIG=./configs/config-rn34.yml ./openai-cli run model.inference
OPENAI_CONFIG=./configs/config-b0.yml ./openai-cli run model.inference
OPENAI_CONFIG=./configs/config-dense161-dropout.yml ./openai-cli run model.inference
OPENAI_CONFIG=./configs/config-inceptionv3.yml ./openai-cli run model.inference
#CUDA_VISIBLE_DEVICES=0 OPENAI_CONFIG=./configs/config-inceptionrnv2.yml ./openai-cli run model.train
