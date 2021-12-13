import face_recognition as fr
import cv2
from face_recognition.api import face_encodings
import numpy as np
import pickle
import os
from flask import Flask, render_template, Response
app=Flask(__name__)

def load_data():
    file = open("data.pkl", "rb")
    encoded = pickle.load(file)
    file.close()
    return encoded

def unknown_image_encoded(img):
    """
    encode a face given the file name
    """
    face = fr.load_image_file("faces/" + img)
    encoding = fr.face_encodings(face)[0]

    return encoding


def main(im):
    """
    will find all of the faces in a given image and label
    them if it knows what they are

    :param im: str of file path
    :return: list of face names
    """
    print("Start")
    faces = load_data()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    print("Reading image")
    img = cv2.imread(im, 1)
    #img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    #img = img[:,:,::-1]
    face_locations = fr.face_locations(img)
    unknown_face_encodings = fr.face_encodings(img, face_locations)
    width = int(img.shape[1] * 50 / 100)
    height = int(img.shape[0] * 50 / 100)
    dim = (width, height)
    
    print("Recognition")
    face_names = []
    for face_encoding in unknown_face_encodings:
        # See if the face is a match for the known face(s)
        
        matches = fr.compare_faces(faces_encoded, face_encoding)
        name = "Unknown"

        # use the known face with the smallest distance to the new face
        face_distances = fr.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a box around the face
            cv2.rectangle(img, (left-20, top-20), (right+20, bottom+20), (255, 0, 0), 2)

            # Draw a label with a name below the face
            cv2.rectangle(img, (left-20, bottom -15), (right+20, bottom+20), (255, 0, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, name, (left -20, bottom + 15), font, 1.0, (255, 255, 255), 2)

    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    # Display the resulting image
    print("Show Face Recognition")
    while True:

        # cv2.imshow('Video', resized)
        ret, buffer = cv2.imencode('.jpg', resized)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        dim = (width, height)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
# def start_fr():
#     # testPath = ""
#     # for dirpath, dnames, fnames in os.walk("./test face"):
#     #     for f in fnames:
#     #         if f.endswith(".jpg") or f.endswith(".png"):
#     #             testPath = "test face\\" + f
    
#     facePath = "./image/face.jpg"
#     classify_face(facePath)