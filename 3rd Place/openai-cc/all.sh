OPENAI_CONFIG=./configs/base-config.yml ./openai-cli run tools.prepare_target
OPENAI_CONFIG=./configs/base-config.yml ./openai-cli run tools.make_folds
OPENAI_CONFIG=./configs/base-config.yml ./openai-cli run tools.extend_test

OPENAI_CONFIG=./configs/config-dense121.yml ./openai-cli run model.inference
OPENAI_CONFIG=./configs/config-rn34.yml ./openai-cli run model.inference
OPENAI_CONFIG=./configs/config-b0.yml ./openai-cli run model.inference
OPENAI_CONFIG=./configs/config-dense161-dropout.yml ./openai-cli run model.inference
OPENAI_CONFIG=./configs/config-inceptionv3.yml ./openai-cli run model.inference

OPENAI_CONFIG=./configs/base-config.yml ./openai-cli run model.blend
OPENAI_CONFIG=./configs/base-config.yml ./openai-cli run tools.add_missed
