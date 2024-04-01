
from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline
import mediapipe as mp
import cv2
import numpy as np
import base64
from datetime import datetime

from minio import Minio
from io import BytesIO
import uuid
import json
# Configure MinIO client
minioClient = Minio(
    "localhost:9000",  # Change to your MinIO server address
    access_key="minioadmin",  # Change to your access key
    secret_key="minioadmin",  # Change to your secret key
    secure=False  # Set to True if using HTTPS
)


def upload_to_minio(bucket_name, file_path, content, content_type):
    try:
        minioClient.put_object(
            bucket_name,  # Bucket name
            file_path,  # Object name including the constructed directory structure
            data=BytesIO(content),  # The file content
            length=len(content),
            content_type=content_type  # MIME type
    )
    except Exception as e:
        print(f"Failed to upload to MinIO: {e}")

now = datetime.now()
dir_path = f"year{now.year}/year={now.year}/month={now.month}/day={now.day}/hour={now.hour}/minute={now.minute}/{uuid.uuid4()}/{uuid.uuid4()}/"


def detect_face_direction(image_path, face_mesh, save_directory='annotated_images'):
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

            # Determine the direction
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

            if direction == "Forward":
                # Convert the annotated image to bytes
                _, img_bytes = cv2.imencode('.jpg', image)
                img_bytes = img_bytes.tobytes()

                # Encode the image bytes to Base64 for embedding in JSON
                base64_image = base64.b64encode(img_bytes).decode('utf-8')

                # Create JSON content with the Base64-encoded image
                json_content = {
                    "image_base64": base64_image,
                    "direction": direction,
                    "angles": {"x": x, "y": y, "z": z}
                }

                # Convert the JSON content to bytes
                json_bytes = json.dumps(json_content).encode('utf-8')

                # Define the file path for the image and JSON within the same directory
                image_file_path = f"{dir_path}image.jpg"
                json_file_path = f"{dir_path}response.json"

                # Upload the image to MinIO
                image_upload_success = upload_to_minio("faceapi-session", image_file_path, img_bytes, "image/jpeg")

                # Upload the JSON to MinIO
                try:
                    minioClient.put_object(
                        "faceapi-session",  # Bucket name
                        json_file_path,  # JSON file path
                        data=BytesIO(json_bytes),  # JSON content
                        length=len(json_bytes),
                        content_type='application/json'  # MIME type
                    )
                    json_upload_success = True
                except Exception as e:
                    print(f"Failed to upload JSON to MinIO: {e}")
                    json_upload_success = False

                # Construct the paths to return
                image_save_path = f"minio/faceapi-session/{image_file_path}" if image_upload_success else None
                json_save_path = f"minio/faceapi-session/{json_file_path}" if json_upload_success else None

                return {
                    "direction": direction,
                    "angles": {"x": x, "y": y, "z": z},
                    "saved_image_path": image_save_path,
                    "saved_json_path": json_save_path
                }


app = Flask(__name__)

UPLOAD_FOLDER = 'received_videos'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/receive_video_frames', methods=['POST'])
def receive_video_frames():
    video_frames_binary = request.files['video_frames'].read()

    # Construct the file path based on the current datetime and a unique ID
    file_path = f"{dir_path}video.mp4"

    # Upload the video to MinIO
    upload_success = upload_to_minio("faceapi-session", file_path, video_frames_binary, "video/mp4")

    message = 'Video frames received and saved to MinIO' if upload_success else 'Failed to save video frames to MinIO'
    return jsonify({'message': message})



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
    return render_template('index11.html')

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
            print(result)
            predictions.append(float(result))

    average_prediction = sum(predictions) / len(predictions) if predictions else 0
    response_message = "Real" if average_prediction > 0.1 else "Fake"

    return jsonify({
        "message": response_message,
        "average_prediction": average_prediction,
        "directions": directions  # Include the detailed face directions in the response
    })


if __name__ == "__main__":
    if os.path.exists('cert.pem') and os.path.exists('key.pem'):
        app.run(host='0.0.0.0', port=8001, ssl_context=('cert.pem', 'key.pem'))
    else:
        app.run(host='0.0.0.0', port=8001)
