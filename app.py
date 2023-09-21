from flask import Flask, render_template, request, redirect, url_for
import cv2
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import numpy as np
import dlib
import math
import base64

app = Flask(__name__)

@app.route('/')
def upload_page():
    return render_template('upload.html')

@app.route('/preview', methods=['POST'])
def preview_page():
    id_card_image = request.files['idCard']
    personal_image = request.files['personalPhoto']
    
    # Save the uploaded images temporarily (you can modify this part)
    id_card_image_path = 'static/' + id_card_image.filename
    personal_image_path = 'static/' + personal_image.filename
    id_card_image.save(id_card_image_path)
    personal_image.save(personal_image_path)
    
    return render_template('preview.html', id_card_image=id_card_image_path, personal_image=personal_image_path)



from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ... (previous imports and route definitions)

@app.route('/confusion_matrix')
def confusion_matrix_page():
    # Provided image paths
    id_card_images = ['static\id1.jpg', 'static\id2.jpg', 'static\id3.jpg','static\id4.jpg',
                      'static\id6.jpg', 'static\id6.jpg', 'static\id5.jpg', 'static\id4.jpg',
                      'static\id3.jpg','static\id2.jpg']
    person_images = ['static\id1.jpg', 'static\id2.jpg', 'static\id3.jpg','static\id4.jpg',
                      'static\id6.jpg', 'static\id6.jpg', 'static\id5.jpg', 'static\id4.jpg',
                      'static\id3.jpg','static\id2.jpg']

    # Initialize lists for true labels and predicted labels
    true_labels = []
    predicted_labels = []

    # Load the pre-trained Haar Cascade Classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Iterate through the image pairs
    for id_card_path, person_path in zip(id_card_images, person_images):
        id_card_image = cv2.imread(id_card_path)
        person_image = cv2.imread(person_path)

        # Convert images to grayscale
        id_card_gray = cv2.cvtColor(id_card_image, cv2.COLOR_BGR2GRAY)
        person_gray = cv2.cvtColor(person_image, cv2.COLOR_BGR2GRAY)

        # Perform face detection using cascade classifier
        id_card_faces = face_cascade.detectMultiScale(id_card_gray, scaleFactor=1.1, minNeighbors=5)
        person_faces = face_cascade.detectMultiScale(person_gray, scaleFactor=1.1, minNeighbors=5)

        # Determine the true label (you need to set this based on your data)
        # For example, if the images are matches, set true_label to 1, else set to 0
        true_label = 1 if 'id1' in id_card_path else 0

        # Determine the predicted label based on face detection results
        predicted_label = 1 if len(id_card_faces) > 0 and len(person_faces) > 0 else 0

        true_labels.append(true_label)
        predicted_labels.append(predicted_label)

    # Create the confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    return render_template('confusion_matrix.html', conf_matrix=conf_matrix)

@app.route('/process_images', methods=['POST'])
def process_images():
    id_card_image = request.files['id_card_image']
    person_image = request.files['person_image']

    # Convert the uploaded files to OpenCV images
    id_card_image = cv2.imdecode(np.fromstring(id_card_image.read(), np.uint8), cv2.IMREAD_COLOR)
    person_image = cv2.imdecode(np.fromstring(person_image.read(), np.uint8), cv2.IMREAD_COLOR)

    # Load the pre-trained Haar Cascade Classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert images to grayscale
    id_card_gray = cv2.cvtColor(id_card_image, cv2.COLOR_BGR2GRAY)
    person_gray = cv2.cvtColor(person_image, cv2.COLOR_BGR2GRAY)

    # Perform face detection using cascade classifier
    id_card_faces = face_cascade.detectMultiScale(id_card_gray, scaleFactor=1.1, minNeighbors=5)
    person_faces = face_cascade.detectMultiScale(person_gray, scaleFactor=1.1, minNeighbors=5)

    # Determine the true label (you need to set this based on your data)
    # For example, if the images are matches, set true_label to 1, else set to 0
    true_label = 1 if len(id_card_faces) > 0 and len(person_faces) > 0 else 0

    # Determine the predicted label based on face detection results
    predicted_label = 1 if len(id_card_faces) > 0 and len(person_faces) > 0 else 0

    # Create the confusion matrix
    conf_matrix = confusion_matrix([true_label], [predicted_label])

    return render_template('confusion_matrix.html', conf_matrix=conf_matrix)






    
    
@app.route('/next_page', methods=['POST'])
def next_page():
    id_card_image = request.form['id_card_image']
    personal_image = request.form['personal_image']
    
    # Load the pre-trained Haar Cascade Classifier for face detection
    # face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    
    # Load and convert the images to grayscale
    personal_image_cv2 = cv2.imread(personal_image)
    id_card_image_cv2 = cv2.imread(id_card_image)
    
    personal_gray = cv2.cvtColor(personal_image_cv2, cv2.COLOR_BGR2GRAY)
    id_card_gray = cv2.cvtColor(id_card_image_cv2, cv2.COLOR_BGR2GRAY)
    
    # Perform face detection for personal image
    personal_faces = face_cascade.detectMultiScale(personal_gray, scaleFactor=1.1, minSize=(30, 30))
    
    
    # Drawing rectangles around the detected faces in personal image
    for (x, y, w, h) in personal_faces:
        cv2.rectangle(personal_image_cv2, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Perform face detection for ID card image
    id_card_faces = face_cascade.detectMultiScale(id_card_gray, scaleFactor=1.1, minSize=(30, 30))
    
    # Drawing rectangles around the detected faces in ID card image
    for (x, y, w, h) in id_card_faces:
        cv2.rectangle(id_card_image_cv2, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Convert images from OpenCV format to PIL format
    personal_pil_image = Image.fromarray(cv2.cvtColor(personal_image_cv2, cv2.COLOR_BGR2RGB))
    id_card_pil_image = Image.fromarray(cv2.cvtColor(id_card_image_cv2, cv2.COLOR_BGR2RGB))

    # Convert images to base64 to embed them directly in the HTML
    import io
    import base64
    
    personal_buffered = io.BytesIO()
    personal_pil_image.save(personal_buffered, format="JPEG")
    personal_img_base64 = base64.b64encode(personal_buffered.getvalue()).decode()

    id_card_buffered = io.BytesIO()
    id_card_pil_image.save(id_card_buffered, format="JPEG")
    id_card_img_base64 = base64.b64encode(id_card_buffered.getvalue()).decode()
    
    # Extracted face images
    extracted_personal_faces = []# To store individual face images from personal image
    
    extracted_id_card_faces = []  # To store individual face images from ID card image
    
    # Drawing rectangles around the detected faces and extracting face images
    for idx, (x, y, w, h) in enumerate(personal_faces):
        personal_face_cropped = personal_image_cv2[y:y+h, x:x+w]
        # Convert the extracted face to base64
        _, personal_face_buffer = cv2.imencode('.jpg', personal_face_cropped)
        personal_face_base64 = base64.b64encode(personal_face_buffer).decode()
        extracted_personal_faces.append(personal_face_base64)
        
        

        
    
    for idx, (x, y, w, h) in enumerate(id_card_faces):
        id_card_face_cropped = id_card_image_cv2[y:y+h, x:x+w]
        # Convert the extracted face to base64
        _, id_card_face_buffer = cv2.imencode('.jpg', id_card_face_cropped)
        id_card_face_base64 = base64.b64encode(id_card_face_buffer).decode()
        print(id_card_face_base64)
        extracted_id_card_faces.append(id_card_face_base64)
        # Only get the first index

        
    
        # Create a list to store comparison results and similarity scores
    comparison_results = []
    similarity_threshold = 0.75
    # Loop through the extracted faces and compare them
    for personal_face_base64, id_card_face_base64 in zip(extracted_personal_faces, extracted_id_card_faces): 
    
        # Decode the base64 images
        personal_face_bytes = base64.b64decode(personal_face_base64)
        id_card_face_bytes = base64.b64decode(id_card_face_base64)

        # Convert bytes to numpy arrays for processing
        personal_face_np = np.frombuffer(personal_face_bytes, np.uint8)
        id_card_face_np = np.frombuffer(id_card_face_bytes, np.uint8)

        # Decode the numpy arrays to images
        personal_face_cv2 = cv2.imdecode(personal_face_np, cv2.IMREAD_GRAYSCALE)
        id_card_face_cv2 = cv2.imdecode(id_card_face_np, cv2.IMREAD_GRAYSCALE)

        # Resize the cropped face to the size of the reference image
        id_card_face_resized = cv2.resize(id_card_face_cv2, (personal_face_cv2.shape[1], personal_face_cv2.shape[0]))

        # Calculate SSIM (Structural Similarity Index)
        similarity_score = ssim(personal_face_cv2, id_card_face_resized, win_size=3)
        print(similarity_score)

        # Compare the similarity score against the threshold
        if similarity_score > 0.70:
            match_status = "ID-Card and Image is matched!!!"
        else:
            match_status = " ID-Card and Image is not  matched!!!"

        # Append comparison results to the list along with match status
        comparison_results.append((personal_face_base64, id_card_face_base64, similarity_score, match_status))

    # Enumerate the comparison_results list
    enumerated_comparison_results = list(enumerate(comparison_results, start=1))
    
    personal_image = request.form['personal_image']
    
    # Load the image using OpenCV
    image = cv2.imread(personal_image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over each detected face
    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) corresponding to the detected face
        face_roi = gray[y:y+h, x:x+w]
        # Load the pre-trained facial landmarks detector
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        # Detect facial landmarks for the ROI
        landmarks = predictor(face_roi, dlib.rectangle(0, 0, w, h))

        # Extract the coordinates of pupils (landmarks 36 and 45)
        x1, y1 = landmarks.part(36).x, landmarks.part(36).y  # Left pupil
        x2, y2 = landmarks.part(45).x, landmarks.part(45).y  # Right pupil
        # Calculate the distance between pupils in pixels
        distance_pixels = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        distance_mm = distance_pixels / 4.1
        distance_mm = round(distance_mm, 2)
        
        # Draw rectangles around eyes
        cv2.rectangle(image, (x + x1 - 10, y + y1 - 10), (x + x1 + 10, y + y1 + 10), (0, 255, 0), 2)
        cv2.rectangle(image, (x + x2 - 10, y + y2 - 10), (x + x2 + 10, y + y2 + 10), (0, 255, 0), 2)

    # Convert the modified image to base64
    _, img_encoded = cv2.imencode('.jpg', image)
    image_base64 = base64.b64encode(img_encoded).decode('utf-8')


    return render_template('next_page.html', 
        personal_image=personal_img_base64, 
        id_card_image=id_card_img_base64, 
        extracted_personal_faces=extracted_personal_faces,
        extracted_id_card_faces=extracted_id_card_faces,
        enumerated_comparison_results=enumerated_comparison_results,
        distance_mm=distance_mm,
        personal_images=image_base64,
        comparison_results=comparison_results)
# ... (rest of the app)





if __name__ == '__main__':
    app.run(debug=True)




