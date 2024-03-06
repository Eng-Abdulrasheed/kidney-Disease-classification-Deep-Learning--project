from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)

clApp = ClientApp()  # Move this outside to ensure it's initialized globally

@app.route("/", methods=['GET'])
def home():
    return render_template('index4.html')

@app.route("/predict", methods=['POST'])
def predictBatchRoute():
    images = request.json.get('images', [])  # Safely get images list
    predictions = []

    for base64_image in images:
        decodeImage(base64_image, clApp.filename)
        result = clApp.classifier.predict()
        if result is not None:  # Check if result is not None
            predictions.append(float(result))  # Assuming result is a single value, convert to float directly

    # Calculate average of predictions if there are any, else set to 0
    average_prediction = sum(predictions) / len(predictions) if predictions else 0

    # Determine the response based on the average_prediction value
    response_message = "Real" if average_prediction > 0.3 else "Fake"
    # Inside your Flask app
    return jsonify({
        "message": response_message,
        "average_prediction": average_prediction  # Add this line to include the numerical average in the response
    })


if __name__ == "__main__":
    if os.path.exists('cert.pem') and os.path.exists('key.pem'):
        app.run(host='0.0.0.0', port=8004, ssl_context=('cert.pem', 'key.pem'))
    else:
        app.run(host='0.0.0.0', port=8004)
