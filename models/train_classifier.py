# import libraries
import sys
from sqlalchemy import create_engine
import re
import numpy as np
import pandas as pd
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import pickle

def load_data(database_filepath):
    '''
    Input:
    database_filepath (str): Database filepath
    
    Output:
    X (DataFrame): Message input data
    Y (DataFrame): Classification target data
    category_names (list(str)): List of columns  
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('df',engine) # read in table
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = list(Y.columns) 
    return X, Y, category_names

def tokenize(text:str) -> list:
    '''
    Input:
    text (str): Raw text
    
    Output
    clean_tokens (list(str)): Cleaned and tokenized text
    '''
    # Normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Tokenize text
    tokens = word_tokenize(text)
    # Lemmatize and remove stop words
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens

def build_model():
    '''
    Input:
    None
    
    Ouptut:
    cv: Gridsearch containing the optimal parameters
    '''
    pipeline = Pipeline([
        ('features', FeatureUnion([
            
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),
        ('moc', MultiOutputClassifier(estimator=RandomForestClassifier()))
    ])
   
    # Define parameters grid
    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'moc__estimator__n_estimators': [5, 15],
    }

    # Create grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Input: 
    model: Trained model
    X_test: X Test dataset
    Y_test: Y Test dataset
    category_names: Names for classification
    
    Output:
    Classification report
    '''
   # Predictions on test data
    Y_pred = model.predict(X_test)

    # Iterate through each category and print classification report
    for i, category in enumerate(Y_test.columns):
        print("Category:", category)
        print(classification_report(Y_test[category], Y_pred[:, i]))
                         
    return True


def save_model(model, model_filepath):
    '''
    Input:
    model: Trained model
    model_filepath: Model filepath
    
    Output: 
    Model as a pickle file
    '''
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)
    return True


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