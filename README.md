# Disaster Response Pipeline Project

### Overview
This project reads in Disaster message csv files from Figure 8, clean and restructure the data, and then export the data into a database.
Then read from the database fit the data to a model to predict what type of message it is.  Then, deploy the model in a web app that will allow an end
user to type in a message, and see what the expected response would be.

The intent of this design is to allow any updates to the Figure 8 dataset to run through the process (provided the struture of the data remains constant).

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.
- To run ETL pipeline that cleans data and stores in database
    `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
- To run ML pipeline that trains classifier and saves
    `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`


2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
3. Go to http://0.0.0.0:3001/  (or localhost:3301)
