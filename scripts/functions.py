# Imports
import re
import numpy as np
import emoji
import pickle
import spacy
import es_core_news_md
from nltk.stem.snowball import SnowballStemmer
from spellchecker import SpellChecker

spacy.prefer_gpu()
nlp = es_core_news_md.load()

# Cargamos diccionario de stopwords
SPANISH_STOPWORDS = pickle.load(open('scripts/stopwords_es', 'rb'))
EMOJI_SENTIMENT_DICT = pickle.load(open('scripts/emoji_dict', 'rb'))

# Creamos diccionario para corregir palabras
spell = SpellChecker(language='es', distance=1)

# Scripts

def remove_punctuation_space(df):
    """Eliminamos signos de puntuación sin
    sustituir por espacio."""

    PUNCTUATION = re.compile("""(\..)|(\...)|(\....)|(\.....)|(\......)|(\.......)""")

    return " ".join([PUNCTUATION.sub(" ", word.lower()) for word in df.split()])


def remove_punctuation(df):
    """Eliminamos signos de puntuación y
    sustituimos por espacio"""

    PUNCTUATION = re.compile("""(\.)|(\;)|(\:)|(\!)|(\?)|(\,)|(\¡)|(\¿)| \
    (\")|(\()|(\))|(\[)|(\])|(\d+)|(\/)|(\“)|(\”)|(\')|(\-)|(\")|(\*)""")

    return " ".join([PUNCTUATION.sub("", word.lower()) for word in df.split()])


def fix_abbr(x):
    """Corrección de palabras abreviadas"""
    if type(x) == list:
        words = x
    elif type(x) == str:
        words = x.split()
    else:
        raise TypeError('El formato no es válido, debe ser lista o str')

    abbrevs = {'d': 'de',
               'x': 'por',
               'xa': 'para',
               'as': 'has',
               'q': 'que',
               'k': 'que',
               'dl': 'del',
               'xq': 'porqué',
               'dr': 'doctor',
               'dra': 'doctora',
               'sr': 'señor',
               'sra': 'señora',
               'm': 'me'}
    return " ".join([abbrevs[word] if word in abbrevs.keys() else word for word in words])

def remove_links(df):
    """Sustituimos links por '{link}'"""
    return " ".join(['{link}' if ('http') in word else word for word in df.split()])


def remove_repeated_vocals(df):
    """Script para eliminar vocales repetidas
    de las palabras."""
    list_new_word = []

    for word in df.split():  # separamos en palabras
        new_word = []
        pos = 0

        for letra in word:  # separamos cada palabra en letras
            # print(word, letra, pos, '-', new_word)
            if pos > 0:
                if letra in ('a', 'e', 'i', 'o', 'u') and letra == new_word[pos - 1]:
                    None
                else:
                    new_word.append(letra)
                    pos += 1
            else:
                new_word.append(letra)

                pos += 1
        else:
            list_new_word.append("".join(new_word))

    return " ".join(list_new_word)


def normalize_laughts(df):
    """Normaliza risas en 5 tipos"""
    list_new_words = []
    for word in df.split():  # separamos en palabras
        count = 0
        vocals_dicc = {'a': 0, 'e': 0, 'i': 0, 'o': 0, 'u': 0}

        for letra in word:
            # print(word)
            if letra == 'j':
                count += 1
            if letra in vocals_dicc.keys():
                vocals_dicc[letra] += 1
        else:
            if count > 3:
                dicc_risa = {'a': 'jaja', 'e': 'jeje', 'i': 'jiji', 'o': 'jojo', 'u': 'juju'}
                risa_type = max(vocals_dicc, key=lambda x: vocals_dicc[x])  # Indica si es a,e,i,o,u
                list_new_words.append(dicc_risa[risa_type])
            else:
                list_new_words.append(word)

    return " ".join(list_new_words)


def remove_hashtags(df):
    """Sustituimos hashtag por {hash}"""
    return " ".join(['{hash}' if word.startswith('#') else word for word in df.split()])


def remove_mentions(df):
    """Sustituimos menciones por {mencion}"""
    return " ".join(['{menc}' if word.startswith('“@') or word.startswith('@') else word for word in df.split()])



def transform_icons(df):
    """Función para identificar los 'emojis'
    tradicionales escritos mediante carácteres"""

    word_list = []
    pos_emojis = [':)', ':D', ':))', ':)))', 'xD', 'xd', 'XD']
    neg_emojis = [':(', ":'(", '>:(', ':,(', ":(("]
    for word in df.split():
        if word in neg_emojis:
            word = '{emoji_neg}'
            word_list.append(word)
        elif word in pos_emojis:
            word = '{emoji_pos}'
            word_list.append(word)
        elif ':O' in word:
            word = '{emoji_neu}'
            word_list.append(word)
        else:
            word_list.append(word)
    return " ".join(word_list)

def remove_eur(df):
    """Script para identificar divisas"""
    wlist = ['{eur}' if ('€' in word) | ('euro' in word) | ('$' in word) else word for word in df.split()]
    return " ".join(wlist)


def sep_emojis(df):
    """Separamos emojis que vengan juntos"""
    words_list = []
    for token in df.split():
        new_word = []
        for letra in token:
            if letra in emoji.UNICODE_EMOJI: #emoji.UNICODE_EMOJI['es'] para version en castellano
                words_list.append(letra)
            else:
                new_word.append(letra)
        else:
            words_list.append("".join(new_word))

    return (" ".join(word for word in words_list if word != ''))


## Scripts para el manejo de emojis en texto

"""def _build_dict_from_csv(csv_path):
    #Crea un diccionario de emojis.
    #Fork basado en emosent-py de Fintel Labs Inc.

    emoji_sentiment_rankings = {}

    with open(csv_path, newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        _header_row = next(csv_reader)
        for row in csv_reader:
            emoji = row[0]
            unicode_codepoint = row[1]
            occurrences = int(row[2])
            negative = float(row[4])
            neutral = float(row[5])
            positive = float(row[6])

            emoji_sentiment_rankings[unicode_codepoint] = {
                'emoji': emoji,
                'occurrences': occurrences,
                'sentiment': [negative, neutral, positive]
            }

    return emoji_sentiment_rankings"""


#EMOJI_SENTIMENT_DICT = _build_dict_from_csv('data/Emoji_Sentiment_Data_v1.0.csv')

def char_is_emoji(character):
    """Devuelve true/false si es emoji"""
    try:
        return character in emoji.UNICODE_EMOJI['es']
    except:
        return character in emoji.UNICODE_EMOJI


def get_emoji_rank(emoji):
    """Devuelve si un emoji es neg/pos/neutro o no identificado"""
    emoji = f'0x{ord(emoji[0]):X}'.lower()

    if emoji in EMOJI_SENTIMENT_DICT.keys():
        if EMOJI_SENTIMENT_DICT[emoji]['occurrences'] > 80:
            if np.argmax(EMOJI_SENTIMENT_DICT[emoji]['sentiment']) == 0:
                return '{emoji_neg}'
            elif np.argmax(EMOJI_SENTIMENT_DICT[emoji]['sentiment']) == 1:
                return '{emoji_neu}'
            else:
                return '{emoji_pos}'
        else:
            return '{emoji_neu}'
    else:
        return '{emoji_neu}'


def transform_emoji(df):
    """Transforma texto con emoji en el label de ese emoji:
    {emo_pos}, {emo_neu}, {emo_neg}"""
    return " ".join([get_emoji_rank(word) if char_is_emoji(word) else word for word in df.split()])

## Scripts para el manejo de texto

def remove_stopwords(df):
    """Script para eliminar stopwords del texto"""
    return " ".join([word for word in df.split() if word not in SPANISH_STOPWORDS])

def correcting_words(df):
#    """Script para corregir palabras mal escritas"""
    misspelled = spell.unknown(df.split())
    return " ".join([spell.correction(word) if word in misspelled else word for word in df.split()])

def lemmatizer(df):
    #Lematización de palabras en castellano
    word_list = []
    doc = nlp(df)
    for tok in doc:
        if str(tok) == 'menc':
            word_list.append('{menc}')
        elif str(tok) == 'hash':
            word_list.append('{hash}')
        elif str(tok) == 'link':
            word_list.append('{link}')
        elif str(tok) == 'emoji_pos':
            word_list.append('{emoji_pos}')
        elif str(tok) == 'emoji_neu':
            word_list.append('{emoji_neu}')
        elif str(tok) == 'emoji_neg':
            word_list.append('{emoji_neg}')
        else:
            # if str(tok) != '{' or str(tok) != '}':
            word_list.append(tok.lemma_.lower())

    return " ".join([word for word in word_list if (word != '{') and (word != '}')])


# Stemmizamos
def stem(df):
    stemmer = SnowballStemmer('spanish')
    return " ".join([stemmer.stem(word) for word in df.split()])



# Main function
def transform_tweets(df, mode='lemma'):
    """ Put it all together"""
    df = remove_links(df)
    df = remove_punctuation_space(df)
    df = remove_mentions(df)
    df = remove_hashtags(df)
    df = remove_eur(df)
    df = transform_icons(df)
    df = sep_emojis(df)
    df = transform_emoji(df)
    df = normalize_laughts(df)
    df = remove_punctuation(df)
    df = remove_repeated_vocals(df)
    df = correcting_words(df)
    df = fix_abbr(df)
    df = remove_stopwords(df)
    if mode == 'lemma':
        df = lemmatizer(df)
    elif mode == 'stem':
        df = stem(df)
    else:
        raise TypeError('Invalid mode. Must be "lemma" or "stem"')

    return df
