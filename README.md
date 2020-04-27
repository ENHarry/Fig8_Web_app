# Udacity's Disaster Response Pipeline Project

This is the 3rd project from the DS Nanodegree
The data was provided by Figure 8, and the goal is to classify messages sent in disasters, based on the need.

## Content
This repo contains the following files:
* An ETL Pipeline for cleaning the data, located in the [data folder](./data/process_data.py)
* A machince learning pipeline for building and training the Sklearn Ada Boost Classifier located in the [models folder](./models/train_classifier.py)
* The flask web app file.

## Execution Steps
1. Download the repo, and in your terminal navigate to the repo directory.
2. Ensure you have python and all the relevant packages installed and then run the following in exact order.
       - 'python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db'
       - 'python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl'
3. Then navigate to the app directory and run 'python run.py' 
4. After running follow [link](https://view6914b2f4-3001.udacity-student-workspaces.com/) to view.
