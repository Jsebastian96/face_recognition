# app.py
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import face_recognition
import numpy as np
import cv2
import base64
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# Cargar las imágenes de los estudiantes
known_face_encodings = []
known_face_names = []

for filename in os.listdir('student_images'):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image = face_recognition.load_image_file(f'student_images/{filename}')
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(filename.split('.')[0])

@socketio.on('frame')
def handle_frame(data):
    # Decodificar la imagen base64
    img_data = base64.b64decode(data)
    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Procesar la imagen
    unknown_face_encodings = face_recognition.face_encodings(frame)

    if len(unknown_face_encodings) > 0:
        unknown_face_encoding = unknown_face_encodings[0]
        results = face_recognition.compare_faces(known_face_encodings, unknown_face_encoding)
        if True in results:
            first_match_index = results.index(True)
            name = known_face_names[first_match_index]
            emit('recognized', {'name': name})
            return

    emit('recognized', {'name': 'Unknown'})

if __name__ == '__main__':
    socketio.run(app, debug=True)
