
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 22 21:11:50 2021

@author: kushal
"""

import cv2
import numpy as np
import dlib
from imutils import face_utils
import face_recognition
from keras.models import load_model
from statistics import mode
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

#check webcam
USE_WEBCAM = False # If false, loads video file source

# parameters for loading data and images
emotion_model_path = './models/emotion_model.hdf5'
emotionabc = get_labels('fer2013')

# hyper-parameters for bounding boxes shape
frameabc = 10
emotion_windowx = (20, 40)

# loading models
detector = dlib.get_frontal_face_detector()
emotion_classifier = load_model(emotion_model_path)


# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []

# starting video streaming

cv2.namedWindow('kushal')
video_capture = cv2.VideoCapture(0)


# Select video or webcam feed
abc= input("1 to use webcam or 2 to use video")
cap = None
if (abc == "1"):
    cap = cv2.VideoCapture(0) # Webcam source
else:

    #cap = cv2.VideoCapture('./test/testvdo.mp4') # Video file source containing MODI AND OBAMA
    #cap = cv2.VideoCapture('./test/abc.mp4') # TITANIC
    #cap = cv2.VideoCapture('./test/def.mp4') # CROWD
    cap = cv2.VideoCapture('./test/kushal.mp4') # KUSHAL RANGA SELF CLIP





while cap.isOpened(): # True:
    ret, bgr_image = cap.read()

    #bgr_image = video_capture.read()[1]

    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    faces = detector(rgb_image)

    for face_coordinates in faces:

        x1, x2, y1, y2 = apply_offsets(face_utils.rect_to_bb(face_coordinates), emotion_windowx)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        #start preprocessing on the RGB image
        gray_face = preprocess_input(gray_face, True)
        #numpy function to expand the array
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        #loading the emotion prediction on gray_face
        emotion_prediction = emotion_classifier.predict(gray_face)
        print(emotion_prediction)
        emotion_probability = np.max(emotion_prediction)
        
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotionabc[emotion_label_arg]
               
        a_file = open("sample.txt", "a")
        print(emotion_text, file=a_file)
        
        emotion_window.append(emotion_text)

        if len(emotion_window) > frameabc:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((0, 0,255))
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
        else:
            color = emotion_probability * np.asarray((0, 255, 0))

        color = color.astype(int)
        color = color.tolist()

        draw_bounding_box(face_utils.rect_to_bb(face_coordinates), rgb_image, color)
        draw_text(face_utils.rect_to_bb(face_coordinates), rgb_image, emotion_mode,
                  color, 0, -45, 1, 1)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('Kushal_mini_project', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
