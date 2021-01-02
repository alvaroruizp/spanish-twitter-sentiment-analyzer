
# Imports
import nltk
from nltk.corpus import stopwords
import spacy
import es_core_news_md

spacy.prefer_gpu()
nlp = es_core_news_md.load()

# Creamos diccionario de stopwords
nltk.download('stopwords')
spanish_stopwords = stopwords.words('spanish')
non_stopwords = ['no', 'ni', 'poco', 'mucho', 'nada', 'muchos', 'muy', 'nosotros',
                 'nosotras', 'vosotros', 'vosotras', 'ellos', 'ellas', 'ella', 'él', 'tu', 'tú', 'yo',
                 'pero', 'hasta', 'contra', 'por']
spanish_stopwords = [word for word in stopwords.words('spanish') if word not in non_stopwords]

from scripts.functions import transform_tweets

def predict_tweet(tweet):
    return transform_tweets(tweet)