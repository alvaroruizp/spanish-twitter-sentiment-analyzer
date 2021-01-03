# Spanish Twitter Sentiment Analyzer
Análisis de sentimiento para tweets en español mediante NLP y machine learning.

### Puedes verlo en producción en https://twittersentiment-es.herokuapp.com/

## Contenido:
- Notebook detallado del proceso de desarrollo.
- Scripts para ejecutar el modelo entrenado sin necesitar acudir al Notebook.


## Descripción:
El propósito de este proyecto es desarrollar un modelo capaz de predecir el sentimiento de tweets en Español basándonos en un algoritmo de machine learning.

Para el entrenamiento, se ha empleado un dataset con más de 240.000 tweets labelizados en sentimiento negativo, neutro y positivo proporcionado por SEPLN (Spanish Society for Natural Language Processing). También se ha añadido otro set de datos clasificados del que han sido empleados aproximadamente 45.000 tweets.

Para el tratamiento de los datos se han elaborado script que reducen al mínimo el contenido de los tweets intentando preservar el máximo de información. Las transformaciones que se han realizado han sido:

- Eliminación de signos de puntuación
- Supresión de emojis por las etiquetas {emoji_neg}, {emoji_neu}, {emoji_pos}
- Normalización de expresiones que indicasen risa por 'jajaja', 'jejeje', 'jijiji', 'jojojo', 'jujuju'. No han sido revertidas todas a la misma expresión porque el significado de cada una puede ser completamente distinto.
- Eliminación de stopwords, haciendo cambios en las stopwords predefinidas y dejando intactas las negaciones y algunos pronombres personales que pueden indicar cercanía.
- Supresión de hashtags, links y menciones por {hash}, {link} y {menc}.
- Lematización de todas las palabras con la librería Spacy.

En la construcción de modelos se han optado por redes neuronales de capas GRU, RNN con capas LSTM, RNN bidireccionales y un modelo de red neuronal convolucional 1D. También se han valorado mediante la técnica de CountVectorizer e TFIFD la regresión logística, random forest y SVM como clasificadores. El algoritmo que mejor accuracy ha obtenido ha sido la regresión logística mediante Count Vectorizer, con más del 76% de acierto.

Finalmente, el modelo funciona bien sin tener que haber recurrido a la hiperparametrización, lo que indica que el rendimiento podría ser incluso mejor. Aun así, dada la baja calidad del dataset, el modelo llega a confundir con frecuencia entre negativo-neutro y positivo-neutro, no así entre tweets negativo-positivo. Estas limitaciones intentarán solventarse en próximas iteraciones del modelo.

Para cualquier cosa, no dudes en contactarme.

¡Gracias!
