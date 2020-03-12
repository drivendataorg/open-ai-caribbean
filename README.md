[<img src='https://s3.amazonaws.com/drivendata-public-assets/logo-white-blue.png'>](https://www.drivendata.org/)
<br><br>
<div align="center">
<img src='https://s3.amazonaws.com/drivendata-public-assets/castries_ortho-cog-thumbnail.png' alt='Banner Image' width='500'>
</div>

# Open AI Caribbean Challenge: Mapping Disaster Risk from Aerial Imagery
## Goal of the Competition

Natural hazards like earthquakes, hurricanes, and floods can have a devastating impact on the people and communities they affect. This is especially true where houses and buildings are not up to modern construction standards, often in poor and informal settlements. While buildings can be retrofit to better prepare them for disaster, the traditional method for identifying high-risk buildings involves going door to door by foot, taking weeks if not months and costing millions of dollars.

The World Bank Global Program for Resilient Housing and WeRobotics teamed up to prepare aerial drone imagery of buildings across the Caribbean annotated with characteristics that matter to building inspectors.

In this challenge, the goal was to use aerial imagery to classify the roof material of identified buildings in St. Lucia, Guatemala, and Colombia. Roof material is one of the main risk factors for earthquakes and hurricanes and a predictor of other risk factors, like building material, that are as not readily seen from the air. Machine learning models that are able to most accurately map disaster risk from drone imagery will help drive faster, cheaper prioritization of building inspections and target resources for disaster preparation where they will have the most impact.


## What's in this Repository
This repository contains code volunteered from leading competitors in the [Open AI Caribbean Challenge: Mapping Disaster Risk from Aerial Imagery](https://github.com/drivendataorg/open-ai-caribbean) DrivenData challenge.

#### Winning code for other DrivenData competitions is available in the [competition-winners repository](https://github.com/drivendataorg/competition-winners).


## Winning Submissions

Place |Team or User | Public Score | Private Score | Summary of Model
--- | --- | --- | --- | ---
1 | The team | 0.332897 | 0.354327 |Our approach is based on a two-layer pipeline, where the first layer is a CNN image classifier and the second one a GBM model adding extra features to the first layer predictions. The first layer CNNs were trained over a stratified 8 k-folds split scheme. We average the predictions of four slightly different models to improve the accuracy. The second layer add features from location, polygon characteristics and neighbors. Two models, catboost and lightgbm, are blended to improve final accuracy. All the features for both models are absolutely the same, so the only difference between these 2 models is the framework which was used to train them.
2 | ZFTurbo | 0.355551 | 0.368139 | The solution is divided into the following stages. 1) Image extraction: For every house on each map its RGB image is extracted with 100 additional pixels from each side as well as mask based on POLYGON description in GEOJSON format. 2) Metadata extraction and find neighbors data: In the next stage we extract meta information for each house in train and test dataset and several statistics on the material of roofs of neighboring houses for a given house. Features from these files used later in the second level models in addition to neural net predictions. It’s needed because neural nets only see the roof image and related polygon. But neural net doesn’t know the location of house on the overall map. Meta features as well as some neighbor statistics helps to make predictions more precise. 3) Train different convolutional neural nets: This is the longest and most time-consuming stage of calculations. At this stage, 7 different models of neural networks are trained, which differ in following: the type of neural network, the number of folds, different set of augmentations, some training procedure differences, different input shape, etc. 4) Second level models: Predictions from all neural networks, as well as metadata and data for neighbors obtained in the previous steps are further used as input for second-level models. Since the dimension of the task is small, I used three different GBM modules - LightGBM, XGBoost and CatBoost. Each of them starts several times with random parameters. And the predictions are then averaged. The result of each model is a submit file, in the same format as the Leader Board. 5) Final ensemble of 2nd level models: Predictions from each GBM classifier then averaged with the same weight. And the final prediction is generated.
3 | Roof is on fire | 0.365941 | 0.383384 | I trained a few models for image classification. Basically I thought that some standard approach should perform well. So we used multiple models and then blended the predictions. Good image augmentations were pretty useful as usual (I made a few experiments with different compositions of augmentations). Adding DropOut to DenseNet161 increased its performance for this problem.
Bonus | nxuan | 0.473615 | 0.494093 | First I set this problem as a classification problem. The goal to is classify the materials of the roof as more accurate as possible. For data preparation, I segment out each roofs and save into individual images with their mask. The reason to have a mask file is to let the model focus more on the predicting roof. For training, I used transfer learning of resnet18, resnet50, and resnet101. The final submission file is the ensembled average of the result from the three models.

#### [Interview with winners](http://drivendata.co/blog/open-ai-caribbean-winners.html)

#### Benchmark Blog Post: ["Mapping Disaster Risk From Aerial Imagery - Benchmark"](http://drivendata.co/blog/disaster-response-roof-type-benchmark/)
