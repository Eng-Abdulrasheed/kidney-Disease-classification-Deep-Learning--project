from flask import Flask, render_template, request, jsonify
import os
from datetime import datetime

app = Flask(__name__)

UPLOAD_FOLDER = 'received_videos'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('webRTC.html')

@app.route('/receive_video_frames', methods=['POST'])
def receive_video_frames():
    video_frames_binary = request.files['video_frames'].read()
    # Generate a unique filename based on timestamp
    filename = datetime.now().strftime("%Y%m%d%H%M%S") + '.mp4'
    # Save video frames to a directory
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    with open(filepath, 'wb') as f:
        f.write(video_frames_binary)
    return jsonify({'message': 'Video frames received and saved'})

if __name__ == '__main__':
    # Create the directory if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
