# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 08:26:09 2024

@author: Taniya Baral
"""

import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
import pyttsx3

# Initialize the text-to-speech engine
engine = pyttsx3.init()

def speech(text):
    print(text)
    engine.say(text)
    engine.runAndWait()

# Initialize video capture
video = cv2.VideoCapture(0)
labels = []

while True:
    ret, frame = video.read()
    if not ret:
        break

    bbox, label, conf = cv.detect_common_objects(frame)
    output_image = draw_bbox(frame, bbox, label, conf)
    
    cv2.imshow("Object Detection", output_image)
    
    for item in label:
        if item not in labels:
            labels.append(item)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and destroy all windows
video.release()
cv2.destroyAllWindows()

# Construct the sentence for speech
if labels:
    if len(labels) == 1:
        sentence = f"I found a {labels[0]}"
    else:
        sentence = "I found " + ", ".join(f"a {label}" for label in labels[:-1]) + f", and a {labels[-1]}."
    speech(sentence)
else:
    speech("No objects were detected.")
