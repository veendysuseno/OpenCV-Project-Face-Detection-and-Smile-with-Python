import cv2
import os

# Load pre-trained classifiers for face and smile detection
cascade_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cascade_smile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

def detect(gray, frame):
    # Detect faces in the grayscale image
    face = cascade_face.detectMultiScale(gray, scaleFactor=1.29, minNeighbors=6)
    
    for (x, y, w, h) in face:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(frame, 'face', (x, y - 10), font, 1, (255, 255, 0), 2)
        
        # Region of interest for smile detection
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        
        # Detect smiles in the region of interest
        smile = cascade_smile.detectMultiScale(roi_gray, scaleFactor=1.3, minNeighbors=25)
        
        for (sx, sy, sw, sh) in smile:
            # Draw rectangle around the smile
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 255), 2)
            cv2.putText(roi_color, 'smile', (sx, sy - 10), font, 1, (0, 255, 255), 2)
    
    return frame

# Path to the directory containing images
image_directory = 'img/'

# Process each image file in the directory
for filename in os.listdir(image_directory):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Load image
        image_path = os.path.join(image_directory, filename)
        gambar = cv2.imread(image_path)
        if gambar is None:
            print(f"Error loading image: {filename}")
            continue
        
        ubahKeGray = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)

        # Perform face and smile detection
        result = detect(ubahKeGray, gambar)

        # Display the output
        cv2.imshow(f'Face and Smile Detection - {filename}', result)
        
        # Wait until a key is pressed
        key = cv2.waitKey(0)

        # Exit if 'e' is pressed
        if key == ord('e'): # Holf "e" keyboard  --> for exit
            print("Exiting...")
            break

cv2.destroyAllWindows()

# This code is used to detect faces and smiles 
# if there is more than 1 photo or image 
# to be detected with OpenCV
