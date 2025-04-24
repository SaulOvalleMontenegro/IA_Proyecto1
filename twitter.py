from flask import Flask, render_template, request, jsonify
from naivebayes import predict_sentiment 
import time

app = Flask(__name__)

@app.route('/')
def index():

    return render_template('index.html') 

@app.route('/procesar', methods=['POST'])
def procesar():
    data = request.get_json()
    texto = data.get('texto', '')
    
    if not texto or texto.strip() == '':
        return jsonify({'resultado': 'Por favor, ingrese texto válido.'})
    

    inicio = time.time()

    prediccion = predict_sentiment(texto) 
    fin = time.time()
    tiempo_ejecucion = int((fin - inicio) * 1000) 
    

    if prediccion in ["Error: Modelo no entrenado", "Entrada inválida"]:
         resultado = f'Error: {prediccion}'
    else:

        resultado = f'Sentimiento predicho: {prediccion.capitalize()}. Tiempo de ejecución: {tiempo_ejecucion} ms'

    return jsonify({'resultado': resultado})

if __name__ == '__main__':

    app.run(debug=True)