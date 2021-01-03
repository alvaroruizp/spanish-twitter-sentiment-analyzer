
# Imports
from scripts.functions import transform_tweets
import numpy as np



def get_label_predict(probas, i=0.5):
    """Script para variar el threshold de la predicción."""
    if np.argmax(probas)==0:
        return 0
    elif np.argmax(probas)==2:
        return 4
    elif np.argmax(probas)==1 and probas[1]<i:
        if probas[2] > probas[0]:
            return 3
        else:
            return 1
    else:
        return 2


def predict_tweet(tweet, tokenizer, model):
    """Script para predecir"""
    tweet = transform_tweets(tweet, mode="lemma") # O mode='stem' para stemización
    print('Predict_tweet:', tweet)

    sentiment = ['Negativo', 'Neutro-negativo', 'Neutro', 'Neutro-positivo', 'Positivo']
    X_pred = tokenizer.transform([tweet])

    print(model.predict_proba(X_pred))
    return {'proba': list(model.predict_proba(X_pred)[0]),
            'sentiment': sentiment[get_label_predict(model.predict_proba(X_pred)[0])]}