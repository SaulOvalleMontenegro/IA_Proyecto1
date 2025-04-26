# **README - SentimentCheck**   

**Mini Twitter Analyzer**  
Aplicación web para análisis de sentimiento (positivo/negativo/neutral) usando Naive Bayes, Flask y JavaScript.

---

## **Instalación**  

### **Requisitos**  
- Python 3.8+  
- pip  
- Git (opcional)  

### **Pasos**  
1. Clonar repositorio:  
   ```bash
   git clone https://github.com/SaulOvalleMontenegro/IA_Proyecto1.git
   cd SentimentCheck
   ```

2. Instalar dependencias:  
   ```bash
   pip install flask pandas scikit-learn
   ```

3. Ejecutar la app:  
   ```bash
   python app.py
   ```
   > La app estará en `http://localhost:5000`.

---

## **Uso**  
1. **Interfaz web**:  
   - Ingresa texto en el cuadro y haz clic en **"Analizar Sentimiento"**.
   - Recibirás una predicción (Positivo/Negativo/Neutral) y tiempo de procesamiento.

2. **API (para desarrolladores)**:  
   ```bash
   curl -X POST http://localhost:5000/procesar -H "Content-Type: application/json" -d '{"texto":"Me encanta este juego!"}'
   ```
   **Respuesta**:  
   ```json
   {"resultado": "Sentimiento predicho: Positive. Tiempo: 120 ms"}
   ```

---
# Diagrama de Arquitectura (Mermaid)

```mermaid
graph TD
    A[Frontend\nHTML/CSS/JS] -->|POST /procesar| B[Backend\nFlask/Python]
    B -->|predict_sentiment()| C[Modelo\nNaiveBayes]
    C -->|Carga| D[(Dataset\ntwitter_training.csv)]
    C -->|Usa| E[(Stopwords)]
    B -->|JSON Response| A

    classDef front fill:#E1F5FE,stroke:#0288D1
    classDef back fill:#E8F5E9,stroke:#388E3C
    classDef model fill:#FFF8E1,stroke:#FFA000
    classDef data fill:#F5F5F5,stroke:#616161,dashed

    class A front
    class B back
    class C model
    class D,E data
```
---

## **Arquitectura**  
- **Frontend**: HTML/CSS/JS (interfaz simple).  
- **Backend**: Flask (API REST).  
- **Modelo**: Clasificador Naive Bayes con preprocesamiento de texto.  

---

## **Estructura de Archivos**  

```
SentimentCheck/
├── app.py                # Backend Flask (rutas y lógica API)
├── naivebayes.py         # Modelo Naive Bayes (entrenamiento/clasificación)
├── static/
│   └── estilos.css       # Estilos CSS
├── templates/
│   └── index.html        # Interfaz web
└── twitter_training.csv  # Dataset de entrenamiento
```

---

## **Tecnologías**  

| **Tecnología**       | **Uso**                              |
|----------------------|--------------------------------------|
| Python 3             | Lenguaje principal (backend/modelo)  |
| Flask                | Framework web (API REST)             |
| Naive Bayes          | Modelo ML para clasificación         |
| JavaScript (Fetch)   | Comunicación frontend-backend        |
| Pandas               | Procesamiento del dataset            |
| HTML/CSS             | Interfaz de usuario                  |

---

## **Descripción del Proyecto**  
- **Objetivo**: Clasificar texto en sentimientos (ej: reseñas de juegos).  
- **Precisión**: ~80% (dependiendo del dataset).  
- **Limitaciones**:  
  - Solo inglés (por stopwords predefinidas).  
  - Requiere reentrenamiento para nuevos dominios.  
