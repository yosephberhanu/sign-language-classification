import os
import numpy as np
import pandas as pd
import cv2
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import RMSprop,Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates

def label_area(image, start, end, text):
    image = cv2.rectangle(image, start, end, (36,255,12), 1)
    return cv2.putText(image, text, (start[0], start[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

def bounding_box(points):
    x_buffer = 40
    y_buffer = 40
    x_coordinates, y_coordinates = zip(*points)
    return [(min(x_coordinates) - x_buffer, min(y_coordinates) - y_buffer), (max(x_coordinates) + x_buffer, max(y_coordinates) + y_buffer)]

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

model = models.load_model('model')

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
image_size = 100
video_capture = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while True:
        ret, image = video_capture.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Flip the image horizontally for a selfie-view display.
        image = cv2.flip(image,1)
        image_height, image_width, _ = image.shape
        results = hands.process(image)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                points = bounding_box([_normalized_to_pixel_coordinates(h.x, h.y,image_width,image_height) for h in hand_landmarks.landmark if _normalized_to_pixel_coordinates(h.x, h.y,image_width,image_height)])
                roi = image[points[0][1]:points[1][1], points[0][0]: points[1][0]]
                roi=np.array(cv2.resize(roi,(image_size, image_size)))
                # print(.shape)
                image = label_area(image, points[0], points[1], str(np.argmax(model.predict(np.array([roi])))))        
        cv2.imshow('MediaPipe Hands', image[:, :, ::-1])

        key = cv2.waitKey(1) 
        #if ESC is pressed, exit loop
        if key == 27:
            break
    video_capture.release()
    cv2.destroyAllWindows()