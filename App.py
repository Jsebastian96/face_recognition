import face_recognition
import cv2
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)

# Cargar imágenes de estudiantes y codificar sus caras
known_face_encodings = []
known_face_names = []

# Aquí se debe cargar las fotos de los estudiantes y sus nombres correspondientes
# Ejemplo:
# student_image = face_recognition.load_image_file("path/to/student_image.jpg")
# student_face_encoding = face_recognition.face_encodings(student_image)[0]
# known_face_encodings.append(student_face_encoding)
# known_face_names.append("Student Name")

@app.route('/recognize', methods=['POST'])
def recognize():
    # Obtener la imagen desde la solicitud
    image = request.files['image'].read()
    npimg = np.frombuffer(image, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Convertir la imagen de BGR a RGB
    rgb_frame = frame[:, :, ::-1]

    # Encontrar todas las caras en la imagen
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

    return jsonify(face_names)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
