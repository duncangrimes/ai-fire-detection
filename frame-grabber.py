import cv2
import time

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(1)

# Set resolution to 1080p
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

last_capture_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    # Capture a frame every 5 seconds
    if current_time - last_capture_time >= 5:
        filename = f"frame_{int(current_time)}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Captured: {filename}")
        last_capture_time = current_time

    # Display the live feed (optional, press 'q' to quit)
    cv2.imshow("Webcam Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()