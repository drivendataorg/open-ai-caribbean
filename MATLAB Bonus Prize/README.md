Open AI Caribbean Challenge
==============================

Use aerial imagery to classify the roof material of identified buildings in St. Lucia, Guatemala, and Colombia.

Step 1 - Extract roof patches for training and testing
1. Install Anconda with python version 3.7
2. After installation, run the following command to install the necessary packages
conda install -c conda-forge rasterio  
conda install -c conda-forge geopandas
pip install opencv-python
3. Place the raw data files under the data folder
4. run python file 'python_create_data.py' inside the src folder to extract the roof patches. 
P.S. I have already include the extracted roof patched for you to use so you can skip the entire step 1 to save some time, but feel free to extract the roof patches yourself. 


Step 2 - Use matlab to train and generate the submission result from testing date
1. Install Matlab 2019a and make sure it has the deep learning toolbox. 
2. Open 'main.m' under src using matlab
3. I have already included the pre-trained models. You can choose to set 'train_indicator' to true to train the models yourself, or simply set it to false to use the pre-trained model to make classification and generate the submission files.
4. Based on the above option, you will run the 'main.m' file using matlab, and at the end, you will have a 'submission.csv' file created.  

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
	│        └── stac      <- The download raw data.
    │
	├── models             <- Trained and serialized models, model predictions, or model summaries
    └── src                <- Source code for use in this project.
    


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
