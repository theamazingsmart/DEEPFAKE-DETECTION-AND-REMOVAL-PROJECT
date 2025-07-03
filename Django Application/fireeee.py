import cv2
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Get the webcam feed.
capture = cv2.VideoCapture(0)

while True:
    # Capture a frame from the webcam.
    ret, frame = capture.read()

    # Resize the image to be at least 224x224 and then crop from the center.
    size = (224, 224)
    image = Image.fromarray(frame)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    cv2.putText(frame, "Class: " + class_name[2:], (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, "Confidence Score: " + str(confidence_score), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("Webcam", frame)

    # Press `q` to quit.
    if cv2.waitKey(1) == ord("q"):
        break

# Release the webcam.
capture.release()
cv2.destroyAllWindows()
