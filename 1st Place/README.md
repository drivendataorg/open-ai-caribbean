# Open AI Caribbean Challenge: Mapping Disaster Risk from Aerial Imagery
This repo includes code of "The team" team which consists of two individuals 
[Daniel_FG](https://www.drivendata.org/users/Daniel_FG/) and
[PermanentPon](https://www.drivendata.org/users/PermanentPon/) for 
[Open AI Caribbean Challenge: Mapping Disaster Risk from Aerial Imagery](https://www.drivendata.org/competitions/58/disaster-response-roof-type/)


## Solution Summary

Our approach is based on  two layer pipeline, where the first layer is a CNN image classifier and the second one a GBM model adding extra features to the first layer predictions. 

The first layer is trained over a stratified 8 k-folds split squeme. We average the predictions of four slightly different models to improve the accuracy. 

The second layer add features from location, polygon characteristics and neighbors. Two models, catboost and lightgbm, are blended to improve final accuracy.


## Requirements/prerequisites to reproduce this solution
You should have `python3` and `pip3` installed. 

Using GPUs is essential to train all CNN models in a decent time frame. We used Nvidia 
Titan X (Pascal) with 12 GB of memory. Training models using GPU with less memory may 
require to change batch_size in src/models. 

Make sure that you have unpacked input data into `data` folder (!NOT included in the repo!) 

## How to reproduce the solution
If you complied with all the prerequisites just run:

`bash train_predict.sh`

It triggers the training and prediction on the test set. 
When the script has finished running you'll find the prediction in `predictions` folder. 
The final submission would be `submission_cat_lgb_009.csv.gz`.

We use identical 8-folds data split across all models. 
We included `data/8Kfolds_201910302225.csv` file with folds split in the repo.
