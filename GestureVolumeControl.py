import cv2
import time
import numpy as np
import HandTrackingModule as htm

cap = cv2.VideoCapture(0)
detector = htm.HandDetector()
pTime = 0

while True:
    success, img = cap.read()
    img = detector.findHands(img)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.imshow('Gesture Volume Control', img)
    cv2.waitKey(1)