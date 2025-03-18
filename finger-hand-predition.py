import os
import cv2
import mediapipe as mp
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow Lite warnings

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Initialize Webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Define Hand and Face Detection Models
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_detection = mp_face.FaceDetection(min_detection_confidence=0.5)

# Finger tip landmarks (Thumb to Pinky)
FINGER_TIPS = [4, 8, 12, 16, 20]

def count_fingers(hand_landmarks):
    """Counts the number of fingers raised."""
    count = 0
    if hand_landmarks:
        landmarks = hand_landmarks.landmark
        for tip in FINGER_TIPS[1:]:  
            if landmarks[tip].y < landmarks[tip - 2].y:
                count += 1
        if landmarks[FINGER_TIPS[0]].x > landmarks[FINGER_TIPS[0] - 1].x:
            count += 1
    return count

prev_time = 0  # For FPS calculation

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))  # Resize for better performance
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Face Detection
    face_results = face_detection.process(rgb_frame)
    if face_results.detections:
        for detection in face_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box

    # Hand Detection
    hand_results = hands.process(rgb_frame)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            x_min, y_min, x_max, y_max = iw, ih, 0, 0
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * iw), int(lm.y * ih)
                x_min, y_min = min(x, x_min), min(y, y_min)
                x_max, y_max = max(x, x_max), max(y, y_max)

            # Draw hand bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # Blue box

            # Count fingers
            finger_count = count_fingers(hand_landmarks)
            cv2.putText(frame, f"Fingers: {finger_count}", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # FPS Calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Face & Hand Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
