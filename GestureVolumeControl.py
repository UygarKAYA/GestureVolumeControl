import cv2
import time
import math
import numpy as np
import HandTrackingModule as htm

cap = cv2.VideoCapture(0)
detector = htm.HandDetector(max_num_hands=1, min_tracking_confidence=0.7)
pTime = 0

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    landmarksList = detector.findLocation(img, draw=False)

    if len(landmarksList) != 0:
        x4, y4 = landmarksList[4][1], landmarksList[4][2]
        x8, y8 = landmarksList[8][1], landmarksList[8][2]
        cx, cy = (x4+x8)//2, (y4+y8)//2

        cv2.circle(img, (x4, y4), 8, (0, 255, 0), cv2.FILLED)
        cv2.circle(img, (cx, cy), 8, (0, 255, 0), cv2.FILLED)
        cv2.circle(img, (x8, y8), 8, (0, 255, 0), cv2.FILLED)
        cv2.line(img, (x4, y4), (x8, y8), (0, 255, 0), 3)

        length = math.hypot(x8-x4, y8-y4)
        if length < 50:
            cv2.circle(img, (cx, cy), 8, (0, 0, 0), cv2.FILLED)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.imshow('Gesture Volume Control', img)
    cv2.waitKey(1)