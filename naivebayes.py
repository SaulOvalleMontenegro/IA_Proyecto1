import re
import math
from collections import defaultdict
import pandas as pd
import random 

STOPWORDS = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just",
    "don", "should", "now"
])

def tokenize_and_clean(text):
    """
    Convierte texto a minúsculas, elimina URLs, menciones, caracteres
    no alfanuméricos, tokeniza y elimina stopwords.
    """
    if not isinstance(text, str):
        return []
    text = text.lower()

    text = re.sub(r'http\S+|www\S+', '', text)

    text = re.sub(r'@\w+', '', text)

    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = text.split()
    tokens = [token for token in tokens if token not in STOPWORDS and len(token) > 1]
    return tokens



class SentimentClassifierNB:
    """Clasificador Naive Bayes Multinomial para análisis de sentimiento."""
    def __init__(self, smoothing_alpha=1.0):
        self.alpha = smoothing_alpha
        self._categories = set()
        self._vocabulary = set()
        self._log_priors = {}
        self._log_likelihoods = {}
        self._category_doc_counts = defaultdict(int)
        self._category_feature_counts = defaultdict(lambda: defaultdict(int))
        self._category_total_features = defaultdict(int)

    def fit(self, training_data):
        print("Iniciando fase de conteo y construcción de vocabulario...")
        total_docs = 0
        for category, document in training_data:
            category = str(category).lower()
            self._categories.add(category)
            self._category_doc_counts[category] += 1
            total_docs += 1

            tokens = tokenize_and_clean(document)
            for token in tokens:
                self._vocabulary.add(token)
                self._category_feature_counts[category][token] += 1
                self._category_total_features[category] += 1
        print(f"Conteo finalizado. Vocabulario: {len(self._vocabulary)} palabras únicas. Categorías: {self._categories}")

        vocab_size = len(self._vocabulary)
        if vocab_size == 0:
            print("Advertencia: El vocabulario está vacío después del preprocesamiento.")

        print("Calculando probabilidades logarítmicas (priors y likelihoods)...")
        for category in self._categories:
            category_count = self._category_doc_counts[category]
            if total_docs == 0: continue
            self._log_priors[category] = math.log(category_count / total_docs)

            total_features_in_cat = self._category_total_features[category]
            denominator = total_features_in_cat + self.alpha * vocab_size
            if denominator == 0: denominator = 1

            self._log_likelihoods[category] = {}
            for word in self._vocabulary:
                word_count_in_cat = self._category_feature_counts[category].get(word, 0)
                likelihood = (word_count_in_cat + self.alpha) / denominator
                self._log_likelihoods[category][word] = math.log(likelihood) if likelihood > 0 else -float('inf')

            unknown_likelihood = self.alpha / denominator
            self._log_likelihoods[category]['<UNK>'] = math.log(unknown_likelihood) if unknown_likelihood > 0 else -float('inf')
        print("Cálculo de probabilidades finalizado.")


    def classify(self, text):
        if not self._categories or not self._log_priors or not self._log_likelihoods:

            return None
        tokens = tokenize_and_clean(text)
        log_scores = {category: self._log_priors.get(category, -float('inf')) for category in self._categories}

        for token in tokens:
            for category in self._categories:
                 if category in self._log_likelihoods:
                     log_likelihood = self._log_likelihoods[category].get(token, self._log_likelihoods[category].get('<UNK>', -float('inf')))
                     log_scores[category] += log_likelihood
                 else:
                     log_scores[category] = -float('inf')

        if not log_scores or all(score == -float('inf') for score in log_scores.values()):
             if self._log_priors:
                return max(self._log_priors, key=self._log_priors.get)
             else:
                return None
        return max(log_scores, key=log_scores.get)




def evaluate_performance(model, test_data):
    """
    Evalúa el rendimiento del modelo en un conjunto de datos de prueba.
    Calcula Accuracy, y Precision, Recall, F1-Score por clase.
    """
    if not test_data:
        print("Error: No hay datos de prueba para evaluar.")
        return None
    if not model or not model._categories:
        print("Error: El modelo no está entrenado o no tiene categorías definidas.")
        return None

    print(f"\nEvaluando modelo con {len(test_data)} muestras de prueba...")
    true_positives = defaultdict(int)
    false_positives = defaultdict(int)
    false_negatives = defaultdict(int)
    total_predictions = 0
    correct_predictions = 0

    all_labels = set(model._categories)
    for true_label, _ in test_data:
        all_labels.add(str(true_label).lower())
    for label in all_labels:
        true_positives[label] = 0
        false_positives[label] = 0
        false_negatives[label] = 0

    for true_label, text in test_data:
        true_label = str(true_label).lower()
        predicted_label = model.classify(text)
        total_predictions += 1

        if predicted_label is None:
            if true_label in all_labels:
                false_negatives[true_label] += 1
            continue

        predicted_label = predicted_label.lower()

        if predicted_label == true_label:
            correct_predictions += 1
            if true_label in all_labels:
                true_positives[true_label] += 1
        else:
            if predicted_label in all_labels:
                false_positives[predicted_label] += 1
            if true_label in all_labels:
                false_negatives[true_label] += 1

    metrics = {}
    if total_predictions > 0:
        metrics['overall_accuracy'] = correct_predictions / total_predictions
    else:
        metrics['overall_accuracy'] = 0.0
    metrics['per_class'] = {}

    print("\n--- Métricas de Desempeño ---")
    print(f"Accuracy General: {metrics['overall_accuracy']:.4f} ({correct_predictions}/{total_predictions} correctas)")
    print("-" * 30)

    for label in sorted(list(all_labels)):
        tp = true_positives[label]
        fp = false_positives[label]
        fn = false_negatives[label]
        support = tp + fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / support if support > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics['per_class'][label] = {
            'precision': precision, 'recall': recall, 'f1_score': f1_score, 'support': support
        }
        print(f"Clase: {label.capitalize()}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1_score:.4f}")
        print(f"  Support:   {support}")
        print("-" * 20)

    return metrics

#lEER ARHIVO CSV

LOCAL_CSV_FILENAME = "twitter_training.csv"


sentiment_model = None
test_data = []
all_documents = []

try:
    print(f"Intentando cargar datos locales desde: {LOCAL_CSV_FILENAME}")

    df = pd.read_csv(
        LOCAL_CSV_FILENAME,
        header=None, 
        encoding='utf-8'
    )
    print(f"Datos crudos cargados: {len(df)} filas.")

    num_columns = df.shape[1]
    index_sentiment = 2
    index_text = 3    
    required_columns = max(index_sentiment, index_text) + 1

    if num_columns < required_columns:
         raise ValueError(f"Error: El archivo CSV no tiene suficientes columnas (necesita al menos {required_columns}). Encontradas: {num_columns}")

    print(f"Seleccionando columnas por índice: {index_sentiment} (sentimiento) y {index_text} (texto)...")
    df = df[[index_sentiment, index_text]] 

    df.columns = ["category", "text_content"]

    df = df.dropna(subset=["category", "text_content"])

    df["text_content"] = df["text_content"].astype(str)
    df["category"] = df["category"].astype(str).str.lower()

    allowed_sentiments = ["positive", "negative", "neutral"] 
    original_count = len(df)
    print(f"Primeras etiquetas únicas encontradas (antes de filtrar): {list(df['category'].unique()[:10])}")
    df = df[df["category"].isin(allowed_sentiments)]
    filtered_count = len(df)
    if filtered_count < original_count:
         print(f"Filtrado: Se eliminaron {original_count - filtered_count} filas con etiquetas no deseadas/inválidas.")

    if filtered_count == 0:
         raise ValueError(f"No quedaron datos después de procesar y filtrar el archivo {LOCAL_CSV_FILENAME} con etiquetas {allowed_sentiments}. Verifica los índices y el contenido del archivo.")

    print(f"Datos procesados listos para usar: {len(df)} muestras.")
    print("Distribución de clases en los datos procesados:")
    print(df['category'].value_counts())
    print("-" * 30)


    all_documents = list(df.itertuples(index=False, name=None))
    random.seed(42)
    random.shuffle(all_documents)
    split_ratio = 0.8
    split_index = int(len(all_documents) * split_ratio)
    train_data = all_documents[:split_index]
    test_data = all_documents[split_index:]
    if not train_data or not test_data:
         raise ValueError("Conjuntos de entrenamiento o prueba vacíos después de la división.")
    print(f"Divididos en {len(train_data)} para entrenamiento y {len(test_data)} para prueba.")
    print("\nIniciando entrenamiento del modelo...")
    sentiment_model = SentimentClassifierNB(smoothing_alpha=1.0)
    sentiment_model.fit(train_data)
    print("Entrenamiento completado.")
    if test_data:
        evaluation_results = evaluate_performance(sentiment_model, test_data)
    else:
        print("No hay datos de prueba para evaluar.")

except FileNotFoundError:
    print(f"Error Fatal: El archivo '{LOCAL_CSV_FILENAME}' no se encontró.")
    print("Asegúrate de que el nombre del archivo sea correcto y esté en la misma carpeta.")
except ValueError as ve:
    print(f"Error Fatal durante el procesamiento de datos: {ve}")
except Exception as e:
    print(f"Ocurrió un error inesperado durante la carga o procesamiento de datos: {e}")
    sentiment_model = None



# 5. FUNCIÓN DE PREDICCIÓN PARA LA APP FLASK 


def predict_sentiment(text_to_analyze):
    """Función wrapper para usar el modelo entrenado desde otra parte (ej: Flask)."""
    if sentiment_model is None:
         return "Error: Modelo no entrenado"
    if not text_to_analyze or not isinstance(text_to_analyze, str):
        return "Entrada inválida"
    if not hasattr(sentiment_model, '_categories') or not sentiment_model._categories:
         return "Error: Modelo no parece estar entrenado correctamente."
    prediction = sentiment_model.classify(text_to_analyze)
    return prediction if prediction else "Inclasificable"



# 6. BLOQUE PRINCIPAL (Ejecución de ejemplo) (Sin cambios)


if __name__ == "__main__":
    print("\n" + "="*50)
    print("Ejecución del script completada.")
    print("El modelo debería estar entrenado si no hubo errores fatales.")
    print("Se mostraron las métricas de evaluación en el conjunto de prueba.")
    print("="*50 + "\n")
    if sentiment_model:
        print("\n--- Pruebas de Predicción Manual ---")
        test_texts = [
            "This game is amazing, graphics are top notch!",
            "I spent hours trying to fix this bug, very frustrating experience.",
            "The patch notes were released today.",
            "The main character's quest log is updated after completing the mission.",
            "This game is not bad, but not great either."
        ]
        for text in test_texts:
            prediction = predict_sentiment(text)

            pred_str = str(prediction)
            if isinstance(prediction, str) and prediction not in ["Error: Modelo no entrenado", "Entrada inválida", "Inclasificable"]:
                 pred_str = prediction.capitalize()

            print(f"Texto: '{text}'\n  -> Predicción: {pred_str}")
            print("-" * 15)
    else:
         print("\nEl modelo no se pudo entrenar, no se pueden realizar predicciones de prueba.")