from flask import Flask, request, render_template, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY') or os.urandom(24)

MODEL_PATH = '/Users/enzo/Documents/efrei/eye_cancer_classifier/model1.hdf5'
model = tf.keras.models.load_model(MODEL_PATH)


noms_des_classes = ['Disease_Risk', 'DR', 'ARMD', 'MH', 'DN', 'MYA', 'BRVO', 'TSLN', 'ERM', 'LS',
                    'MS', 'CSR', 'ODC', 'CRVO', 'TV', 'AH', 'ODP', 'ODE', 'ST', 'AION', 'PT',
                    'RT', 'RS', 'CRS', 'EDN', 'RPEC', 'MHL', 'RP', 'CWS', 'CB', 'ODPM', 'PRH',
                    'MNF', 'HR', 'CRAO', 'TD', 'CME', 'PTCR', 'CF', 'VH', 'MCA', 'VS', 'BRAO',
                    'PLQ', 'HPED', 'CL']

def predict_image(image_path):
    image = Image.open(image_path)
    image = image.resize((356, 536))
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image)
    return predictions

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'erreur': 'Aucun fichier envoyé'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'erreur': 'Aucun fichier sélectionné'})

    if file:
        filename = secure_filename(file.filename)
        if not os.path.exists('static/uploads'):
            os.makedirs('static/uploads')

        file_path = os.path.join('static/uploads', filename)
        file.save(file_path)

        try:
            predictions = predict_image(file_path)
            results = {}

            if predictions is not None and len(predictions[0]) == len(noms_des_classes):
                for i, prob in enumerate(predictions[0]):
                    results[noms_des_classes[i]] = float(prob)
            else:
                return jsonify({'erreur': 'Erreur interne dans les prédictions'})

               # Retourne le template 'result.html' avec les résultats
            return render_template('result.html', predictions=results)

        except Exception as e:
            return jsonify({'erreur': str(e)})

    return jsonify({'erreur': 'Une erreur est survenue lors du téléchargement du fichier'})

@app.route('/results', methods=['POST'])
def results():
    if request.is_json:
        data = request.get_json()
        return render_template('result.html', predictions=data)

    return jsonify({'erreur': 'Format de données non pris en charge'})

if __name__ == 'main':
    app.run(debug=True)