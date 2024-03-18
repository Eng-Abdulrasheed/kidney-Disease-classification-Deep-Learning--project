from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline
import mediapipe as mp
import cv2
import numpy as np

def detect_face_direction(image_path, face_mesh):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = face_mesh.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, _ = image.shape
    face_2d = []
    face_3d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in [33, 263, 1, 61, 291, 199]:
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    if idx == 1:  # Nose tip for projection
                        nose_2d = (x, y)
                        nose_3d = (x, y, lm.z * 3000)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            focal_length = img_w
            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]], dtype=np.float64)
            distortion_matrix = np.zeros((4, 1), dtype=np.float64)

            success, rotation_vec, translation_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, distortion_matrix)
            rmat, _ = cv2.Rodrigues(rotation_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

            x, y, z = angles[0] * 360, angles[1] * 360, angles[2] * 360

            if x > 10:
                if y > 10:
                    direction = "Looking Up Right"
                elif y < -10:
                    direction = "Looking Up Left"
                else:
                    direction = "Looking Up"
            elif x < -10:
                if y > 10:
                    direction = "Looking Down Right"
                elif y < -10:
                    direction = "Looking Down Left"
                else:
                    direction = "Looking Down"
            else:
                if y > 10:
                    direction = "Looking Right"
                elif y < -10:
                    direction = "Looking Left"
                else:
                    direction = "Forward"



            # Return both direction and angles
            return {"direction": direction, "angles": {"x": x, "y": y, "z": z}}

    return {"direction": "Unable to determine", "angles": None}


app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)
        # Initialize the MediaPipe Face Mesh model here
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

clApp = ClientApp()  # Move this outside to ensure it's initialized globally

@app.route("/", methods=['GET'])
def home():
    return render_template('index3.html')

@app.route("/predict", methods=['POST'])
@app.route("/predict", methods=['POST'])

def predictBatchRoute():
    images = request.json.get('images', [])  # Safely get images list
    predictions = []
    directions = []

    for base64_image in images:
        decodeImage(base64_image, clApp.filename)
        direction_info = detect_face_direction(clApp.filename, clApp.face_mesh)  # Call the face direction function
        directions.append(direction_info)
        result = clApp.classifier.predict()
        if result is not None:
            predictions.append(float(result))

    average_prediction = sum(predictions) / len(predictions) if predictions else 0
    response_message = "Real" if average_prediction > 0.3 else "Fake"

    return jsonify({
        "message": response_message,
        "average_prediction": average_prediction,
        "directions": directions  # Include the detailed face directions in the response
    })


if __name__ == "__main__":
    if os.path.exists('cert.pem') and os.path.exists('key.pem'):
        app.run(host='0.0.0.0', port=8002, ssl_context=('cert.pem', 'key.pem'))
    else:
        app.run(host='0.0.0.0', port=8002)
