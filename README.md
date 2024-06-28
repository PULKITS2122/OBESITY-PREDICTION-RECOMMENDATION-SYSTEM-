# OBESITY PREDICTION (RECOMMENDATION SYSTEM)

_Members:_

| Name (First and Last Name)| Matriculation |
| ------ | ------ |
| Pulkit Sharma | 22201100 |
| Aryan Jain | 22107593 |

**Project Title: Chronic Kidney Disease Prediction System**

GIT LINK: https://mygit.th-deg.de/as23727/recommendation_system


# Project Description

A Machine Learning Model to predict OBESITY of a patient. This project uses RandomForestClassifier Model to train on a given dataset. Interactive GUI is developed using PyQt6.

The file [description.txt](description.txt) contains insights and description of each and every variable in dataset.

The dataset used is from https://www.kaggle.com/datasets/muhramasaputra/obesity-based-on-eating-habits-and-physical-cond
# Prerequisites
To run this project on your commputer, you will need the following software and libraries:
1. Python [3.12]
2. numpy
3. pandas
4. matplotlib
5. scikit-learn
6. PyQt6

# Installation
## Usage

- Download this git repository or clone it to your system using following command:
```
git clone https://mygit.th-deg.de/as23727/recommendation_system
```
- Create a Virual enviroment
```
python -m venv venv
source venv/bin/activate 
# on windows: venv\Scripts\activate.bat
```
- Install required python packages from [requirements.txt](requirements.txt) file using following command:
```
pip install -r requirements.txt
```
- Double click and run [main.py](main.py) file to use the prediction model.

# Implementation of Requests
## Requirements and Setup
1. Python Modules: We have listed all the necessary Python modules in the requirements.txt file. This includes PyQt6 for the GUI, Pandas for data handling, and Scikit-learn for machine learning algorithms.
2. Virtual Environment: We have utilized the venv module to create an isolated Python environment. This ensures that our project dependencies are managed efficiently and do not conflict with system-wide Python packages.
3. Data Source: We sourced our data from a freely available dataset, which can be found on platforms such as Kaggle. The data is in CSV format, suitable for import and analysis.

## Data Handling and Analysis
1. Data Import: The application allows users to import data via a menu button. Additionally, data loading is also available directly upon starting the application.
2. Data Analysis with Pandas: We have integrated Pandas methods to provide an overview of the dataset. This includes the use of dataframe.info(), dataframe.describe(), and dataframe.corr() methods.
3. Additional Metrics and Diagrams: Besides the basic Pandas methods, we also implemented other metrics and diagrams to enhance data analysis and visualization.

## User Interaction and Machine Learning
1. Input Widgets: Our application features several input widgets (at least 3, with 2 being different types) that allow users to modify feature variables.
2. Machine Learning Algorithm: We have applied a Scikit-learn training model algorithm, particularly one inspired by Aurélien Géron's work in Chapter 4 of his book.
3. Output Canvas: For data visualization, we created 1 or 2 output canvases. These display the results of data analysis and predictions in a user-friendly manner.
4. Statistical Metrics: At least 3 statistical metrics are displayed based on the input data, providing insights into the dataset's characteristics.
5. Interactive Predictions: The application is designed to be interactive. Changes in input parameters trigger new predictions and visualizations, enhancing the user experience and making the tool more dynamic.

# Contributions

ARYAN JAIN:
1. Worked on Collection and Preprocessing of Dataset using Pandas.
2. Modifying and Preparing Data with use of numpy arrays.
3. Training Working Logistic Regression Model with help of Scikit-learn.

PULKIT SHARMA:
1. Creating a GUI Interface using PyQt6.
2. Getting User-Inputs using different GUI elements.
3. Integrating Logistic Regression model to main file in order to get predictions.
4. Found Data source from kaggle
