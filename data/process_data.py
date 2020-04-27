import sys

# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    insert path to database for saved data extraction
    Args:
        messages_filepath  : (relative) filepath of disaster_messages.csv
        categories_filepath : (relative) filepath of disaster_categories.csv
        
    Returns:
        A merged data frame
    '''
    # loading data
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge data sets
    df = pd.merge(messages, categories, on = "id")
    return df


def clean_data(df):
    '''
    insert path to database for saved data extraction
    Args:
        df  : merged data frame that needs to be cleaned further
        
    Returns:
        cleaned df
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.loc[0,:]

    
    # extract category names
    category_colnames = row.apply(lambda x: x[:-2])

    # rename the columns of `categories`
    categories.columns = category_colnames

    # convert category columns to binary variables
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df = df.drop(columns=['categories'])

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis= 1)

    # drop duplicates
    df = df.drop_duplicates()

    # drop other unnecessary columns
    df = df.drop(columns=['id', 'original'])

    return df


def save_data(df, database_filename):
    '''
    insert path to database for saved data extraction
    Args:
        df : previously cleaned data
        database_filename  : (relative) name for database
        
    Returns:
        X, Y, category_names
    '''
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('cleanedDF', engine, index=False, if_exists='replace')
      


def main():
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