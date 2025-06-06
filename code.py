import cv2
import mediapipe as mp
import numpy as np
import os
from tkinter import *
import webbrowser

# Suppress warnings by setting an environment variable (if the previous method didn’t work)
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Path to images folder
path = 'Imagebasics'
images = []
classNames = []

# Load images and their labels
myList = os.listdir(path)
print("List of Images:", myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

print("Class Names:", classNames)


# Function to find face landmarks
def findLandmarks(image):
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        landmarks = []
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = image.shape
                bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                landmarks.append(bbox)
        return landmarks


# Known image bounding boxes
known_landmarks = [findLandmarks(img)[0] if findLandmarks(img) else None for img in images]

# Set up GUI
root = Tk()
root.geometry("500x200")


# Function to handle face matching
def Facem():
    cap = cv2.VideoCapture(0)

    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        match_found = False
        while cap.isOpened() and not match_found:
            success, img = cap.read()
            if not success:
                break

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_detection.process(img_rgb)

            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, _ = img.shape
                    bbox = (int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h))

                    # Compare with known_landmarks
                    for i, known_bbox in enumerate(known_landmarks):
                        if known_bbox:
                            # Measure difference in bounding box positions and size
                            bbox_diff = np.linalg.norm(np.array(bbox) - np.array(known_bbox))
                            if bbox_diff < 80:  # Adjust this threshold for better matching
                                name = classNames[i].upper()
                                print("Match found:", name)

                                if not match_found:
                                    webbrowser.open_new(
                                        f'file:///C:/Users/SRIKAR/Desktop/harshitha/project images/{name}.txt')
                                    match_found = True

                                # Draw bounding box and label
                                cv2.rectangle(img, (bbox[0], bbox[1]),
                                              (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                                              (0, 255, 0), 2)
                                cv2.putText(img, name, (bbox[0], bbox[1] - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                                break

            cv2.imshow('Webcam', img)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


# Button to initiate face detection
myButton = Button(root, text="Click here to find details", command=Facem)
myButton.pack()

root.mainloop()