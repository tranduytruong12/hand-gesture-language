import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize webcam
camera_found = False
for camera_index in range(3):  # Try indices 0, 1, and 2
    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        print(f"Successfully opened camera with index {camera_index}")
        camera_found = True
        break
    else:
        cap.release()

if not camera_found:
    print("No camera found. Please check your camera connection.")
    exit()

# Set up MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Use video mode for better tracking
hands = mp_hands.Hands(static_image_mode=False,
                       min_detection_confidence=0.3, max_num_hands=1)

# Map class indices to labels - adjust these based on your training data
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D',
               4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K'}


print("Starting gesture recognition. Press 'q' to quit.")

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Only proceed if we have data
        if data_aux and len(data_aux) == 42:  # 21 landmarks × 2 coordinates
            # Draw bounding box
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10

            # Make prediction
            prediction = model.predict([np.asarray(data_aux)])

            try:
                predicted_character = labels_dict[int(prediction[0])]

                # Draw rectangle and prediction
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, predicted_character, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            except KeyError:
                # Handle case where prediction is not in labels_dict
                print(
                    f"Warning: Predicted class {prediction[0]} not in labels dictionary")

    # Add instruction text
    cv2.putText(frame, "Press 'q' to quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Hand Gesture Recognition', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Application closed")
