from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)

# Загружаем DNN-модель
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
faceNet = cv2.dnn.readNet(faceModel, faceProto)

def highlightFace(net, frame, conf_threshold=0.7):
    frameHeight, frameWidth = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append({'x': x1, 'y': y1, 'w': x2 - x1, 'h': y2 - y1})
    return faceBoxes

@app.route('/')
def home():
    return send_file('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    file = request.files['frame'].read()
    npimg = np.frombuffer(file, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Распознаем лица
    faceBoxes = highlightFace(faceNet, frame)
    count = len(faceBoxes)

    print(f"Найдено лиц: {count}")
    return jsonify({'count': count, 'faces': faceBoxes})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)