from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)

# Prepare the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None
frame_size = (640, 480)  # You might need to adjust this
fps = 20

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_frame', methods=['POST'])
def video_frame():
    global out
    data = request.json['image']
    header, encoded = data.split(",", 1)
    decoded = base64.b64decode(encoded)
    nparr = np.frombuffer(decoded, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Initialize the video writer the first time a frame is received
    if out is None:
        out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (img.shape[1], img.shape[0]))
    
    out.write(img)
    return jsonify(success=True)

@app.route('/stop_recording', methods=['GET'])
def stop_recording():
    global out
    if out is not None:
        out.release()
        out = None
    return jsonify(success=True)

if __name__ == '__main__':
    app.run(debug=True, port=5004)
