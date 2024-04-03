# Disaster Response Pipeline Project

This project trains a message classification system using Figure8 data to assist correctly direct messages to the right channels in emergency situations. The trained model is accessible through a web app..

## Packages

To execute this model, the following packages must be installed and imported accordingly:

- sys
- sqlalchemy
- re
- numpy
- pandas
- nltk
- sklearn
- pickle 
- Flask
- json
- plotly

## Project Summary

This project trains its model emergency text data supplied by Figure8 using an ETL pipeline and a ML pipeline. This model can then be accessed using a web app.

**ETL Pipeline** (`data/process_data.py`) completes the following:
- extracts data from both the supplied datasets ('messages' and 'categories')
- merges, cleans and lemmatizes the datasets
- Loads the transformed data to an SQLite database

**ML Pipeline** (`models/train_classifier.py`) then completes the following:
- loads the tranformed data from the SQL database
- splits this data into train and test sets
- trains and optimises a model using a gridsearch
- stores the model in a pickle file

**Web App** (`app`): This takes a take input and displays a table of the resulting message categories.

## Files

- `README.md`: Read me file 
- `data/process_data.py`: Python script with ETL pipeline
- `data/DisasterResponse.db`: Disaster Response Database
- `data/disaster_categories.csv`: Disaster categories csv file
- `data/disaster_messages.csv`: Disaster messages csv file
- `models/train_classifier.py`: Python script with ML pipeline
- `models/classifier.pkl`: Model storage file
- `app/templates/`: HTML Templates used for the web app
- `app/run.py`: Python script to launch web app
  
## Instructions

To use the web app, download the `data`, `models` and `app` folders and all of their contents.

To train the model, run the following commands in a terminal, ensuring you are in the correct directory:
1. `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
2. `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

To run the app, change directory and run the python script as below:
1. `cd app`
2. `python run.py`
Once this has been executed, you can run the app preview and enter any text into the search bar.
