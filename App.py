from flask import Flask
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import face_recognition
import numpy as np
import cv2
import base64
import os
import sys
from pymongo import MongoClient
from dotenv import load_dotenv


load_dotenv()

MONGO_URI = os.getenv('MONGO_URI')
PORT = int(os.getenv('PORT', 5000))

try:
    import face_recognition_models
except ImportError:
    print("Please install `face_recognition_models` with this command before using `face_recognition`:")
    print("pip install git+https://github.com/ageitgey/face_recognition_models")
    sys.exit(1)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

client = MongoClient(MONGO_URI)
db = client['student_management']
students_collection = db['students']

known_face_encodings = []
students_data = []

for student in students_collection.find():
    if 'photo' in student:
        img_data = base64.b64decode(student['photo'])
        np_arr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            encoding = encodings[0]
            known_face_encodings.append(encoding)
            students_data.append(student)

@socketio.on('frame')
def handle_frame(data):
    img_data = base64.b64decode(data)
    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    unknown_face_encodings = face_recognition.face_encodings(frame)

    if len(unknown_face_encodings) > 0:
        unknown_face_encoding = unknown_face_encodings[0]
        results = face_recognition.compare_faces(known_face_encodings, unknown_face_encoding)
        if True in results:
            first_match_index = results.index(True)
            student = students_data[first_match_index]
            emit('recognized', {
                'name': student['name'],
                'email': student.get('email', 'N/A'),
                'course': student.get('course', 'N/A'),
                'other_data': student.get('other_data', 'N/A')
            })
            return

    emit('recognized', {'name': 'Unknown', 'message': 'No student recognized'})

if __name__ == '__main__':
    socketio.run(app, port=PORT, debug=True)
