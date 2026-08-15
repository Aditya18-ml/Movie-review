import pandas as pd

def runeda():

    imdb=pd.read_csv('IMDB Dataset.csv')
    print('--------head-------')
    print(imdb.head())
    print('-----basic info-----')
    print(imdb.info())
    print('-----describe-----')
    print(imdb.describe())

    print(f" imdb dataset shape:{imdb.shape}")
    print('all the columns name ')
    print(imdb.columns)
    print('is null values present inside the dataset')
    print(imdb.isnull().sum())

    print('drop the duplicate rows ')
    imdb=imdb.drop_duplicates(subset=['review'])
    print('drop the null rows ')
    imdb=imdb.dropna(subset=['review','sentiment'])

    print(imdb['sentiment'].value_counts())

    imdb['sentiment']=imdb['sentiment'].map({'positive':1,'negative':0})
    documents=imdb['review']

if __name__== "__main__":
    runeda()