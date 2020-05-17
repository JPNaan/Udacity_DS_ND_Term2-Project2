import sys
import nltk
import re
import pickle

import pandas as pd
import numpy as np

from sqlalchemy import create_engine
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report

nltk.download('punkt')
nltk.download('wordnet')

def load_data(database_filepath):
    '''
    Parameters:  database_filepath is a string representing the file path to the database containing the data
    This function takes in the data from the specified database and returns the X, y and category names
    '''

    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df = pd.read_sql_table("disaster_data", con=engine)

    X = df['message'].values
    y = df.drop(columns = ['id', 'message', 'original'])
    y = pd.get_dummies(y, columns = ['genre'], prefix = 'genre').values

    category_names = pd.get_dummies(df.drop(columns = ['id', 'message', 'original']), columns = ['genre'], prefix = 'genre').columns

    return X, y, category_names

def tokenize(text):
    '''
    This function tokanizes the text of the X values
    '''

    lst = []
    tok = word_tokenize(text)
    lem = WordNetLemmatizer()

    for t in tok:
        clean = lem.lemmatize(t).lower().strip()
        lst.append(clean)

    return lst

def build_model():
    '''
    This function builds a model leveraging a pipeline and grid search
    '''
    pipeline = Pipeline([\
                     ('vect', CountVectorizer(tokenizer=tokenize)),
                     ('tfidf', TfidfTransformer()), ('multiout',MultiOutputClassifier(RandomForestClassifier(n_estimators=100)))
                    ])

    parameters = {
        'vect__max_features': (None, 5000, 15000),
        'tfidf__use_idf': (True, False)
        }

    #model = pipeline
    model = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, verbose=36, cv = 2)

    return model

def evaluate_model(model, X_test, y_test, category_names):
    '''
    This function evaluates the fitted model using classification report
    '''
    y_pred = model.predict(X_test)

    index = 0
    for label in category_names:
        print(label, index)
        print(classification_report(y_test[index], y_pred[index]))
        index = index + 1
    return

def save_model(model, model_filepath):
    '''
    Saves the trained model as a pickle file to the specified filepath for later use.
    '''
    pickle.dump(model, open(model_filepath, 'wb'))
    return

def main():
    '''
    Main executable function that combines the other functions with printable messages for execution
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
        'as the first argument and the filepath of the pickle file to '\
        'save the model to as the second argument. \n\nExample: python '\
        'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()
