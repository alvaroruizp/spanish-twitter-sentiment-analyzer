
import re
from nltk.corpus import stopwords
import spacy
import es_core_news_md

# Especificamos el uso de GPU para Spacy
spacy.require_gpu()
nlp = es_core_news_md.load()

# Sustituimos menciones, hashtags, link y emojis
# Normalizamos risas
# Eliminamos letras repetidas
# Sustituimos signos de puntuacion
# Corregimos abreviaciones
# Eliminamos stopwords
# Lematizamos

#############

# Eliminamos signos de puntuación y sustituimos por espacio
def remove_punctuation_space(df):
    PUNCTUATION = re.compile("""(\..)|(\...)|(\....)|(\.....)|(\......)|(\.......)""")

    return " ".join([PUNCTUATION.sub(" ", word.lower()) for word in df.split()])

# Eliminamos signos de puntuación sin sustituir
def remove_punctuation(df):
    PUNCTUATION = re.compile("""(\.)|(\;)|(\:)|(\!)|(\?)|(\,)|(\¡)|(\¿)| \
    (\")|(\()|(\))|(\[)|(\])|(\d+)|(\/)|(\“)|(\”)|(\')|(\-)|(\")""")

    return " ".join([PUNCTUATION.sub("", word.lower()) for word in df.split()])

# Corregimos abreviaciones
def fix_abbr(x):
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

# Sustituimos links por {link}
def remove_links(df):
    return " ".join(['{link}' if ('http') in word else word for word in df.split()])

# Eliminamos vocales repetidas
def remove_repeated_vocals(df):
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

# Normalizamos risas 'jajaja', 'jejeje', 'jojojo'
def normalize_laughts(df):
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

# Sustituimos hashtag por {hash}
def remove_hashtags(df):
    return " ".join(['{hash}' if word.startswith('#') else word for word in df.split()])

# Sustituimos menciones por {mencion}
def remove_mentions(df):
    return " ".join(['{menc}' if word.startswith('“@') or word.startswith('@') else word for word in df.split()])

# Función para identificar los 'emojis' tradicionales
def transform_icons(df):
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

# Definimos función para identificar sentimiento del emoji
from emosent import get_emoji_sentiment_rank
import emoji

# Devuelve true/false si es emoji
def char_is_emoji(character):
    return character in emoji.UNICODE_EMOJI

# Devuelve si un emoji es neg/pos/neutro o no identificado
def sentiment_emoji(emoji):
    try:
        sentiment = get_emoji_sentiment_rank(emoji)['sentiment_score']
        if sentiment < -0.3:
            return '{emoji_neg}'
        elif sentiment < 0.3:
            return '{emoji_neu}'
        elif sentiment >= 0.3:
            return '{emoji_pos}'
    except:
        return '{emoji_na}'

# Transformamos emojis
def transform_emoji(df):
    return " ".join([sentiment_emoji(word) if char_is_emoji(word) else word for word in df.split()])

# Separamos emojis que vengan juntos
def sep_emojis(df):
    words_list = []
    for token in df.split():
        new_word = []
        for letra in token:
            if letra in emoji.UNICODE_EMOJI:
                words_list.append(letra)
            else:
                new_word.append(letra)
        else:
            words_list.append("".join(new_word))

    return (" ".join(word for word in words_list if word != ''))

# Eliminamos stopwords
def remove_stopwords(df):
    spanish_stopwords = stopwords.words('spanish')
    non_stopwords = ['no', 'ni', 'poco', 'mucho', 'nada', 'muchos', 'muy', 'nosotros',
                     'nosotras', 'vosotros', 'vosotras', 'ellos', 'ellas', 'ella', 'él', 'tu', 'tú', 'yo']
    spanish_stopwords = [word for word in stopwords.words('spanish') if word not in non_stopwords]

    return " ".join([word for word in df.split() if word not in spanish_stopwords])

# Lematizamos
def lemmatizer(df):

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



###
# Put it all together
def transform_tweets(df):
    df = remove_punctuation_space(df)
    df = remove_mentions(df)
    df = remove_links(df)
    df = remove_hashtags(df)
    df = transform_icons(df)
    df = sep_emojis(df)
    df = transform_emoji(df)
    df = normalize_laughts(df)
    df = remove_punctuation(df)
    df = remove_repeated_vocals(df)
    df = fix_abbr(df)
    df = remove_stopwords(df)
    # df = stem(df) #Opción para stemizar
    df = lemmatizer(df)
    return df

import time
start_time = time.time()

# my code here
frases = ['Hoy ha sido un día genial. Tengo muchas ganas de repetirlo',
          'Hoy ha sido un día horrible. No tengo nada de ganas de repetirlo',
          'Hoy ha sido un día genial. Tengo muchas ganas de repetirlo']

for frase in frases:
    print(transform_tweets(frase))

#print(transform_tweets('Hoy ha sido un día genial. Tengo muchas ganas de repetirlo'))

print("time elapsed: {}".format(time.time() - start_time))
