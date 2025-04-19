import os
import cv2
import time

# Directory setup
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 10 
dataset_size = 100

# Try to find an available camera
camera_found = False
for camera_index in range(10):  # Try indices 0, 1, and 2
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

# Create class directories
for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j+1))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j+1}')
    print('Ready? Press "Q" to start collecting images.')

    # Wait for Q press to start collecting
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to grab frame")
            time.sleep(0.1)
            continue

        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    # Collect dataset
    counter = 0
    print(f"Collecting {dataset_size} images for class {j}...")
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to grab frame")
            time.sleep(0.1)
            continue

        cv2.imshow('frame', frame)
        cv2.waitKey(25)

        # Save the image
        img_path = os.path.join(DATA_DIR, str(j), f'{counter}.jpg')
        cv2.imwrite(img_path, frame)

        counter += 1
        if counter % 10 == 0:
            print(f"Collected {counter}/{dataset_size} images")

cap.release()
cv2.destroyAllWindows()
print("Dataset collection complete!")
