import cv2
import mediapipe as mp
import time


class HandDetector:
    def __init__(self, static_image_mode=False, max_num_hands=2, model_complexity=1,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_hads = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.static_image_mode, self.max_num_hads, self.model_complexity,
                                        self.min_detection_confidence, self.min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)

        if self.result.multi_hand_landmarks:
            for handLms in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findLocation(self, img, handNumber=0, draw=True):
        landmarksList = []

        if self.result.multi_hand_landmarks:
            myHands = self.result.multi_hand_landmarks[handNumber]
            for ID, landmark in enumerate(myHands.landmark):
                height, weight, channel = img.shape
                cx, cy = int(landmark.x * weight), int(landmark.y * height)
                landmarksList.append([ID, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 1, (0, 0, 255), cv2.FILLED)

        return landmarksList


def main():
    cap = cv2.VideoCapture(0)
    previousTime = 0

    while True:
        success, img = cap.read()
        detector = HandDetector()
        img = detector.findHands(img)
        landmarks = detector.findLocation(img)

        if len(landmarks) != 0:
            print(landmarks)

        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime

        cv2.putText(img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.imshow("Hand Tracking Basics", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
