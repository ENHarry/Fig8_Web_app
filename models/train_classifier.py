import sys

# import libraries
import nltk
import pickle
import re
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def load_data(database_filepath):
    '''
    insert path to database for saved data extraction
    Args:
        database_filepath  : (relative) filepath of cleanedDF
        
    Returns:
        X, Y, category_names
    '''


    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('cleanedDF',engine)
    X = df['message'].astype(str)
    Y = df.loc[:, df.columns != 'message']
    # drop the genre column in both the training and testing data sets
    Y = Y.drop(columns=['genre'])
    category_names = Y.columns
    return X, Y, category_names

# create contraction dictionary
contractions = {"aren\'t": "are not","can\'t": "cannot","can\'t\'ve": "cannot have","\'cause": "because","could\'ve": "could have", "couldn\'t": "could not","couldn\'t\'ve": "could not have","c\'mon": "come on","didn\'t": "did not","doesn\'t": "does not","don\'t": "do not","hadn\'t": "had not","hadn\'t\'ve": "had not have","hasn\'t": "has not","haven\'t": "have not","he\'d": "he had ","he\'d\'ve": "he would have","he\'ll": "he will","he\'s": "he is","here\'s": "here is","how\'d": "how did","how\'d\'y": "how do you","how\'ll": "how will","how\'s": "how is","i\'d": "i would","i\'d\'ve": "i would have","i\'ll": "i will","i\'ll\'ve": "i will have","i\'m": "i am","i\'ve": "i have", "isn\'t": "is not","it\'d": "it had","it\'d\'ve": "it would have","it\'ll": "it will",
                "it\'ll\'ve": "it will have","it\'s": "it is","let\'s": "let us","gonna": "going to","ma\'am": "madam",
                "mayn\'t": "may not","might\'ve": "might have","mightn\'t": "might not","mightn\'t\'ve": "might not have",
                "must\'ve": "must have","mustn\'t": "must not","mustn\'t\'ve": "must not have","needn\'t": "need not",
                "needn\'t\'ve": "need not have","o\'clock": "of the clock","oughtn\'t": "ought not",
                "oughtn\'t\'ve": "ought not have","shan\'t": "shall not","sha'n\'t": "shall not",
                "shan't\'ve": "shall not have","she\'d": "she had","she\'d\'ve": "she would have","she\'ll": "she will",
                "she\'ll\'ve": "she will have","she\'s": "she is","should\'ve": "should have","shouldn\'t": "should not",
                "shouldn\'t\'ve": "should not have","so\'ve": "so have","so\'s": "so is","that\'d": "that would",
                "that\'d\'ve": "that would have","that\'s": "that is","there\'d": "there had",
                "there\'d\'ve": "there would have","there\'s": "there is","there\'ve": "there have","they\'d": "they had",
                "they\'d\'ve": "they would have","they\'ll": "they will","they\'ll\'ve": "they will have",
                "they\'re": "they are","they\'ve": "they have","to\'ve": "to have","wasn\'t": "was not","we\'d": "we would",
                "we\'d\'ve": "we would have","we\'ll": "we will","we\'ll\'ve": "we will have","we\'re": "we are",
                "we\'ve": "we have","weren\'t": "were not","what\'ll": "what will",
                "what\'ll\'ve": "what will have","what\'re": "what are","what\'s": "what is","what\'ve": "what have",
                "when\'s": "when is","when\'ve": "when have","where\'d": "where did","where\'s": "where is",
                "where\'ve": "where have","who\'ll": "who will","who\'ll\'ve": "who will have","who\'s": "who is",
                "who\'ve": "who have","why\'s": "why is","why\'ve": "why have","will\'ve": "will have","shall": 'will',
                "won\'t": "will not","won\'t\'ve": "will not have","would\'ve": "would have","wouldn\'t": "would not",
                "wouldn\'t\'ve": "would not have","wanna": "want to","y\'all": "you all","y\'all\'d": "you all would",
                "y\'all\'d\'ve": "you all would have","y\'all\'re": "you all are","y\'all\'ve": "you all have",
                "you\'d": "you would","you\'d\'ve": "you would have","you\'ll": "you will","you\'ll\'ve": "you will have",
                "you\'re": "you are","you\'ve": "you have"
               }
contractions_re = re.compile('(%s)' % '|'.join(contractions.keys()))
def expand_contractions(s, cdict=contractions):
    '''
    takes in strng contractions and expands them 
    using the match function from re
    Args:
        s : (relative) string
        cdict : default contraction dictionary
        
    Returns:
        expanded string
    '''
    def replace(match):
        return cdict[match.group(0)]

    return contractions_re.sub(replace, s.lower())



def tokenize(text):
    '''
    
    Args:
        text : takes each row 
        
    Returns:
        cleaned tokens
    '''
    # clear any contractions
    text= expand_contractions(text, cdict=contractions)
    
    # create regex patterns to be removed
    ptn = '\s{1}\-{1}\s{1}|[\(,.:\';)!"?/]|\-{1}\s{1}'

    # substitute pattern with white space
    text = re.sub(ptn, ' ', text)
    
    # tokenize each row
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return clean_tokens


def build_model():
    '''
    
    Args:
        None
        
    Returns:
        Machine Learning Pipeline
    '''
    # create machine learning pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
        ])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    insert path to database for saved data extraction
    Args:
        model  : resulting model from built function
        X_test : testing text
        Y_test : testing categories
        
    Returns:
        None, instead prints results.
    '''
    # predict on test data
    y_pred = model.predict(X_test)

    # iterate through to print report
    i =0
    for col in category_names:
        print(col)
        print('_'*80)
        print(classification_report(Y_test[col], y_pred[:,i]))
        print("Confusion Matrix:\n", confusion_matrix(Y_test[col], y_pred[:,i]))
        print("\n")
        print("Accuracy: {0:.3f}".format((Y_test[col] == y_pred[:,i]).mean()))
        print("\n")
        i += 1


def save_model(model, model_filepath):
    '''
    Args:
        model_filepath  : (relative) filepath of model
        
    Returns:
        None, instead save the trained model
    '''
    # saving model to pickle file
    pickle_path = model_filepath

    pickle_var = open(pickle_path, 'wb')
    pickle.dump(model, pickle_var)
    pickle_var.close()

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
   

        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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