from scripts.functions import transform_tweets
from scripts.predict_tweet import *
import pickle

# Cargamos modelos
tokenizer = pickle.load(open('models/tokenizer.model', 'rb')) # o tokenizer_stem para lematizar
model = pickle.load(open('models/model_lr.model', 'rb')) # o model_lr_stem para lematizar


if __name__ == '__main__':
    print(predict_tweet("""Este va a ser un tweet positivo
    porque el predictor va a acertar
    perfectamente con su clase""", tokenizer=tokenizer, model=model))

