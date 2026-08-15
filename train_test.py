import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from preprocessing import preprocess_text

def trainmodel():
    imdb=pd.read_csv('IMDB Dataset.csv')

    print("-------Preprocessing text data----")
    filter_rows = preprocess_text(imdb['review'])

    X_train,X_test,y_train,y_test=train_test_split(filter_rows,imdb['sentiment'],test_size=0.2,random_state=42)

    print('------TF-IDF model training ------')
    tfidf=TfidfVectorizer(max_features=5000,ngram_range=(1, 2),stop_words='english')
    X_train_new=tfidf.fit_transform(X_train)
    X_test_new=tfidf.transform(X_test)


    logistic=LogisticRegression()
    logistic.fit(X_train_new,y_train)
    logistic_pred=logistic.predict(X_test_new)
    logistic_acc=accuracy_score(y_test,logistic_pred)
    print(f"logistic regression accuracy:{logistic_acc}")
    print(classification_report(y_test,logistic_pred))

    print('--saving the model and tfidf ----')
    joblib.dump(logistic, 'logistic.joblib')
    joblib.dump(tfidf,'tfidf.joblib')
    print('model saved ')

if __name__=="__main__":
    trainmodel()