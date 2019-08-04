# Master of Data Science | Research Project

# Building A Machine Learning To Predict Taxi Trip Duration Using Partial Data

Code and files used in doing my data-science research project at University of Malaya in 2019.

Name: Ammar Alyousfi
Supervisor: Dr. Salimah Mokhtar
Special semester of the academic year 2018/2019

# Description of the Files

## "Report.pdf"

This is the current version of the report of this project. It contains documentation and explanation about all steps of this project.

## "Code Notebooks" Folder

This folder contains the Jupyter notebooks that include the code used in this project with its results. 

* `Data Preparation and Exploration.ipynb` notebook contains code used in chapter 4 of the project for data exploration and preparation.

* `Modeling 1 - Model Selection.ipynb` notebook contains code used in chapter 5 for feature selection, feature engineering, model selection, and hyperparameter optimization.

* `Modeling 2 - Final Model Building.ipynb` notebook contains code used in chapter 5 for building the final ensemble model and  training and testing the final model.

The data files used by these notebooks are included in the `Code Notebooks` folder. But due to the large size of some files, it wasn't possible to upload them to Github. So we included a small sample of them in this repository. To run the code that use those files, you need to get the complete files. Below is a list of those large files and the links to download them from Googel Cloud Storage:

* `yellow_tripdata_2017-03_processed.csv`. To download the complete version of this file, [click here](https://storage.googleapis.com/research_project_um/yellow_tripdata_2017-03_processed.csv). This file is produced by `Data Preparation and Exploration.ipynb` notebook and it is used by `Modeling 1 - Model Selection.ipynb` and `Modeling 2 - Final Model Building.ipynb` notebooks.

* `yellow_tripdata_2018-03_processed.csv`. To download the complete version of this file, [click here](https://storage.googleapis.com/research_project_um/yellow_tripdata_2018-03_processed.csv). This file is produced by `Data Preparation and Exploration.ipynb` notebook and it is used by `Modeling 1 - Model Selection.ipynb` and `Modeling 2 - Final Model Building.ipynb` notebooks.

* `./NYC_taxi_data/yellow_tripdata_2017-03.csv`. To download the complete version of this file, [click here](https://storage.googleapis.com/research_project_um/yellow_tripdata_2017-03.csv). This file was downloaded originally from NYC Taxi and Limousine Commission website. It is used by `Data Preparation and Exploration.ipynb` notebook.

* * `./NYC_taxi_data/yellow_tripdata_2018-03.csv`. To download the complete version of this file, [click here](https://storage.googleapis.com/research_project_um/yellow_tripdata_2018-03.csv). This file was downloaded originally from NYC Taxi and Limousine Commission website. It is used by `Data Preparation and Exploration.ipynb` notebook.

Note: when running the notebooks, you might need to change some of the file paths inside of them to refer to paths on your machine.

## "Flask App" Folder

This folder contains the files of the Flask web application which was built to deploy the model of this project. The app was deployed on Googel App Engine and can be accessed from [this link](https://cohesive-bonbon-246315.appspot.com/).

## "Taxi-trips meta files" Folder

This folder contains some files related to the taxi-trips datasets such as data dictionary file and map files.
