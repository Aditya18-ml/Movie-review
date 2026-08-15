import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')



def clean_text(text):
    text=str(text)
    text=re.sub(r'<.*?>',' ',text)
    text=re.sub(r'https?://\S+',' ',text)
    text=re.sub(r'[^\w\s]',' ',text)
    
    text=re.sub(r'\d+',' ',text)
    text=text.lower()
    return text


def preprocess_text(headlines):
    lemmatizer=WordNetLemmatizer()
    stop_words=set(stopwords.words('english'))
    cleaned_data=[]
    for headline in headlines:
        cleaned=clean_text(headline)
        tokens=word_tokenize(cleaned)
        filtered_tokens=[lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        cleaned_data.append(' '.join(filtered_tokens))

        
    return cleaned_data
