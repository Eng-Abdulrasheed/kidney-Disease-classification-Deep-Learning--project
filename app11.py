from flask import Flask, render_template, request, jsonify
import websocket
import json
from datetime import datetime
import os

app = Flask(__name__)

KMS_WS_URI = "ws://localhost:8888/kurento"
RECORDINGS_DIR = "received_video_frames"

def send_to_kms(message):
    """Send messages to KMS via WebSocket and return the response."""
    with websocket.create_connection(KMS_WS_URI) as ws:
        ws.send(json.dumps(message))
        result = ws.recv()
    return json.loads(result)

@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html')

def start_recording_kms(recording_path):
    # This function should ideally create a MediaPipeline, a WebRtcEndpoint (linked to your video source),
    # and a RecorderEndpoint configured with the recording_path. Then, it starts recording.
    # The actual implementation will vary based on your application's needs.
    
    # Simplified: Send a WebSocket message to KMS to create a MediaPipeline
    media_pipeline_response = send_to_kms({
        "id": 1,
        "jsonrpc": "2.0",
        "method": "create",
        "params": {
            "type": "MediaPipeline",
            "constructorParams": {},
            "properties": {}
        }
    })
    media_pipeline_id = media_pipeline_response.get("result", {}).get("value")
    
    # Further steps would involve creating WebRtcEndpoint and RecorderEndpoint within this pipeline
    # and linking them together. Finally, you would invoke the `record` method on the RecorderEndpoint.
    
    return media_pipeline_id  # This is simplified; you'd typically return the recorder endpoint ID

@app.route('/start_recording', methods=['POST'])
def start_recording():
    """Endpoint to start recording the video stream."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f"{timestamp}.webm"
    # Ensure the recording path is accessible to KMS, especially if running in Docker
    recording_path = f"file://{os.path.join(RECORDINGS_DIR, file_name)}"

    # Attempt to start recording via KMS
    recorder_id = start_recording_kms(recording_path)

    return jsonify({"message": "Recording started", "file": file_name, "recorderId": recorder_id})

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    """Endpoint to stop recording the video stream."""
    # Assuming the recorder ID is sent in the request for simplicity
    recorder_id = request.json.get('recorderId')
    
    # Send a WebSocket message to KMS to stop the recording
    send_to_kms({
        "id": 2,
        "jsonrpc": "2.0",
        "method": "invoke",
        "params": {
            "object": recorder_id,
            "operation": "stop",
            "operationParams": {}
        }
    })

    return jsonify({"message": "Recording stopped"})

if __name__ == '__main__':
    if not os.path.exists(RECORDINGS_DIR):
        os.makedirs(RECORDINGS_DIR)
    app.run(debug=True)
