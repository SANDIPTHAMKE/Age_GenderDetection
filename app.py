import cv2
import matplotlib.pyplot as plt # type: ignore

model = "opencv_face_detector_uint8.pb"
config = "opencv_face_detector.pbtxt"
age_model = "age_net.caffemodel"
age_config = "age_deploy.prototxt"

# Load the model using OpenCV DNN module
net = cv2.dnn.readNetFromTensorflow(model, config)

# Load the pre-trained model using OpenCV's DNN module
age_net = cv2.dnn.readNetFromCaffe(age_config, age_model)
net = cv2.dnn.readNetFromCaffe("gender_deploy.prototxt", "gender_net.caffemodel")

# Function to load models
def load_model(prototxt, model):
    return cv2.dnn.readNet(model, prototxt)

# Load models with error handling
try:
    face_net = load_model("opencv_face_detector.pbtxt", "opencv_face_detector_uint8.pb")
    age_net = load_model("age_deploy.prototxt", "age_net.caffemodel")
    gender_net = load_model("gender_deploy.prototxt", "gender_net.caffemodel")
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

# Load and preprocess image
image = cv2.imread('image.jpg')
if image is None:
    print("Error loading image. Please check the file path.")
    exit()

image = cv2.resize(image, (720, 640))
fr_cv = image.copy()

# Mean values for age and gender predictions
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_labels = ['(0-2)', '(3-6)', '(7-12)', '(13-20)', '(21-25)', '(25-32)', '(33-43)', '(44-53)', '(54-66)', '(67-82)', '(83-100)']
gender_labels = ['Male', 'Female']

# Face detection
fr_h, fr_w = fr_cv.shape[:2]
blob = cv2.dnn.blobFromImage(fr_cv, 1.0, (300, 300), [104, 117, 123], True, False)
face_net.setInput(blob)
detections = face_net.forward()

# Create face bounding boxes
faceBoxes = []
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.7:
        x1 = int(detections[0, 0, i, 3] * fr_w)
        y1 = int(detections[0, 0, i, 4] * fr_h)
        x2 = int(detections[0, 0, i, 5] * fr_w)
        y2 = int(detections[0, 0, i, 6] * fr_h)
        faceBoxes.append([x1, y1, x2, y2])
        cv2.rectangle(fr_cv, (x1, y1), (x2, y2), (0, 255, 0), int(round(fr_h / 150)), 8)

if not faceBoxes:
    print("No face detected")
else:
    # Loop through detected faces
    for faceBox in faceBoxes:
        face = fr_cv[max(0, faceBox[1] - 15):min(faceBox[3] + 15, fr_cv.shape[0] - 1),
                     max(0, faceBox[0] - 15):min(faceBox[2] + 15, fr_cv.shape[1] - 1)]

        # Prepare blob for age and gender prediction
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # Predict gender
        gender_net.setInput(blob)
        genderPreds = gender_net.forward()
        gender = gender_labels[genderPreds[0].argmax()]

        # Predict age
        age_net.setInput(blob)
        agePreds = age_net.forward()
        age = age_labels[agePreds[0].argmax()]

        # Draw predictions
        cv2.putText(fr_cv, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (217, 0, 0), 2, cv2.LINE_AA)

    # Display the image with predictions
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(fr_cv, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()