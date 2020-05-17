import sys
import pandas as pd
import numpy as np
import sqlite3

from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Takes in the filepath for the messages and categories csv files, reads them in, and
    returns a combined dataframe.
    '''
    messages = pd.read_csv('data/disaster_messages.csv')
    categories = pd.read_csv('data/disaster_categories.csv')

    df = messages.merge(categories, on = 'id')

    return df

def clean_data(df):
    '''
    takes in the combined dataframe, cleans the data for processing, and returns
    a cleaned dataframe.
    '''

    categories = df['categories'].str.split(';', expand = True)
    row = categories.iloc[0,:]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    #Convert column values to 0 or 1
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = pd.to_numeric(categories[column])#
    #convert 'related' values to binary with 0 = 0 and (1 or 2) = 1
    categories['related'] = categories['related'].apply(lambda x: 0 if x == 0 else 1)

    #Replace the original categories column in the df with the new
    #category column
    df = df.drop('categories', axis = 1)
    df = pd.concat([df, categories], axis = 1)

    #Remove dups
    df = df.drop_duplicates()

    return df

def save_data(df, database_filename):
    '''
    Takes in the cleaned data, and saves it into a database.
    '''
    fn = 'sqlite:///'+database_filename
    engine = create_engine(fn)
    df.to_sql('disaster_data', engine, index = False, if_exists = 'replace')


def main():
    '''
    Main executable function that combines the other functions with printable messages for execution
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
