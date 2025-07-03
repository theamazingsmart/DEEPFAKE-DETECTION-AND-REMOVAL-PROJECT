import cv2
import numpy as np

def detect_fire(frame):
    # Convert the frame to HSV color space.
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a lower and upper bound for the fire color in HSV.
    lower_bound = np.array([0, 100, 100])
    upper_bound = np.array([10, 255, 255])

    # Threshold the image to identify the fire pixels.
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)

    # Find the contours in the mask.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Use OpenCV's built-in fire detection function to detect fire.
    fire_detection_model = cv2.dnn.readNet("fire_detection_model.xml", "fire_detection_model.bin")
    results = fire_detection_model.detectMultiScale(frame, 1.3, 5)

    # If there are any contours or fire detection results, then there is fire in the image.
    is_fire = len(contours) > 0 or len(results) > 0

    return is_fire

if __name__ == "__main__":
    # Get the webcam feed.
    capture = cv2.VideoCapture(0)

    while True:
        # Capture a frame from the webcam.
        ret, frame = capture.read()

        # Detect fire in the frame.
        is_fire = detect_fire(frame)

        # Display the result.
        if is_fire:
            cv2.putText(frame, "Fire detected!", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "No fire detected.", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Fire Detection", frame)

        # Press `q` to quit.
        if cv2.waitKey(1) == ord("q"):
            break

    # Release the webcam.
    capture.release()
    cv2.destroyAllWindows()
import cv2
import numpy as np

def detect_fire(frame):
    # Convert the frame to HSV color space.
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a lower and upper bound for the fire color in HSV.
    lower_bound = np.array([0, 100, 100])
    upper_bound = np.array([10, 255, 255])

    # Threshold the image to identify the fire pixels.
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)

    # Find the contours in the mask.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Use OpenCV's built-in fire detection function to detect fire.
    fire_detection_model = cv2.dnn.readNet("fire_detection_model.xml", "fire_detection_model.bin")
    results = fire_detection_model.detectMultiScale(frame, 1.3, 5)

    # If there are any contours or fire detection results, then there is fire in the image.
    is_fire = len(contours) > 0 or len(results) > 0

    return is_fire

if __name__ == "__main__":
    # Get the webcam feed.
    capture = cv2.VideoCapture(0)

    while True:
        # Capture a frame from the webcam.
        ret, frame = capture.read()

        # Detect fire in the frame.
        is_fire = detect_fire(frame)

        # Display the result.
        if is_fire:
            cv2.putText(frame, "Fire detected!", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "No fire detected.", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Fire Detection", frame)

        # Press `q` to quit.
        if cv2.waitKey(1) == ord("q"):
            break

    # Release the webcam.
    capture.release()
    cv2.destroyAllWindows()
    
