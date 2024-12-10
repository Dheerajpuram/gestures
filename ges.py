import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Define gestures and detection logic
def detect_gesture(landmarks):
    if not landmarks:
        return None

    gestures = {
        "Open Palm": landmarks[8][1] < landmarks[6][1] and landmarks[12][1] < landmarks[10][1] and landmarks[16][1] < landmarks[14][1] and landmarks[20][1] < landmarks[18][1],
        "Closed Fist": landmarks[8][1] > landmarks[6][1] and landmarks[12][1] > landmarks[10][1] and landmarks[16][1] > landmarks[14][1] and landmarks[20][1] > landmarks[18][1],
        "Thumbs Up": landmarks[4][1] < landmarks[3][1] and landmarks[4][0] > landmarks[3][0],
        "Thumbs Down": landmarks[4][1] > landmarks[3][1] and landmarks[4][0] > landmarks[3][0],
        "Victory Sign": landmarks[8][1] < landmarks[6][1] and landmarks[12][1] < landmarks[10][1] and landmarks[16][1] > landmarks[14][1],
        "OK Sign": abs(landmarks[4][0] - landmarks[8][0]) < 20 and abs(landmarks[4][1] - landmarks[8][1]) < 20,
        "Rock On": landmarks[8][1] < landmarks[6][1] and landmarks[12][1] > landmarks[10][1] and landmarks[20][1] < landmarks[18][1],
        "Call Me": landmarks[4][0] < landmarks[3][0] and landmarks[20][0] > landmarks[18][0],
        "Wave": landmarks[8][1] < landmarks[6][1] and landmarks[12][1] > landmarks[10][1],
        "Pointing": landmarks[8][1] < landmarks[6][1] and landmarks[12][1] > landmarks[10][1] and landmarks[16][1] > landmarks[14][1],
    }

    for gesture, condition in gestures.items():
        if condition:
            return gesture
    return None

# Initialize Video Capture
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame.")
        break

    # Flip the frame for a selfie-view display
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks
            h, w, _ = frame.shape
            landmarks = [[int(lm.x * w), int(lm.y * h)] for lm in hand_landmarks.landmark]

            # Detect gesture
            gesture = detect_gesture(landmarks)
            if gesture:
                print(f"Detected Gesture: {gesture}")
                # Draw the gesture name on the frame
                cv2.putText(frame, gesture, (landmarks[0][0] - 50, landmarks[0][1] - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Hand Gesture Recognition", frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()