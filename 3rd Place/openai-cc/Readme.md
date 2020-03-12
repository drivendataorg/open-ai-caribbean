# Open AI Caribbean Challenge CLI
The roof is on fire (Konstantin Gavrilchik (head), Stepan Konev)

## Usage
```bash
export OPENAI_CONFIG [CONFIG_PATH]
./openai-cli run [COMMAND]
```
Before you start to use the CLI please specify config and add it into the `PATH`.

To reproduce the result, achieved by the team `ROOF IS ON FIRE` one should run the following commands one by one. Make sure that paths in config files are correct for your case. Ubuntu Linux OS is recommended.
The sequence is the following:
* tools.prepare_target
* tools.make_folds
* tools.extend_test

!!!
I have modified run_all.sh for inference only.
Please consider the following file structure. The working directory should contain the following:
* stac folder with dataset
* openai-cc folder with code
* dumps folder with weights
* submission_format.csv
* train_labels.csv
* metadata.csv

In config files, referred in `openai-cc/run_all.sh`  (configs/config-dense121.yml, configs/config-rn34.yml, configs/config-b0.yml, configs/config-dense161-dropout.yml, configs/config-inceptionv3.yml)in the first line you should specify the path to the working directory.
!!!

Then run the file `openai-cc/run_all.sh` for training/inferencing multiple models. For your comfort you can specify `CUDA_VISIBLE_DEVICES` parameter to train them simultaneously. At least one `CUDA_VISIBLE_DEVICES` is required.

Then run
 ```OPENAI_CONFIG=./configs/base-config.yml ./openai-cli run model.blend```
to blend the predictions from the models prepaired before

Finally please run
```OPENAI_CONFIG=./configs/base-config.yml ./openai-cli run tools.add_missed```

The final csv file with `_fixed` post is the one.

----

For simplicity, you can instead run `bash all.sh` which will execute all the commands outlined above.

----

The following commands are also supported:
* eda.stats
* tools.prepare_target
* tools.make_folds
* tools.add_missed
* model.train
* model.inference
* model.evaluate
* visualize.predictions
