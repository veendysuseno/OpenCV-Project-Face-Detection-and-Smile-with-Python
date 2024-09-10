# Face and Smile Detection with OpenCV (for code.py)

This project demonstrates face and smile detection using OpenCV's pre-trained Haar Cascade classifiers. The code detects faces and smiles in images and displays the results with bounding boxes.

## Features

- **Face Detection:** Identifies faces in an image and draws rectangles around them.
- **Smile Detection:** Detects smiles within the detected face regions and highlights them.

## Requirements

- Python 3.x
- OpenCV library (`cv2`)

## Setup

1. **Install OpenCV**: Make sure you have OpenCV installed. You can install it using pip:

    ```bash
    pip install opencv-python
    ```

2. **Download Haar Cascade XML Files**: Ensure the Haar Cascade XML files for face and smile detection are available. OpenCV's pre-trained classifiers are included with the library, so you should not need to download them manually.

## Code

```python
import cv2

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

# Load image
gambar = cv2.imread('img/target-image.jpg')
ubahKeGray = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)

# Perform face and smile detection
result = detect(ubahKeGray, gambar)

# Display the output
cv2.imshow('Face and Smile Detection', result)

# Wait until a key is pressed
key = cv2.waitKey(0)

# Exit if 'e' is pressed
if key == ord('e'):
    print("Exiting...")
    cv2.destroyAllWindows()

# This code will only execute or detect faces and smiles 
# in named photos or images "target-image".

## How to Use
- Place the target image(s) in the img directory.
- Run the script:
```bash
python script_name.py
```

- The image will be displayed with detected faces and smiles highlighted. Press any key to close the image window, or press 'e' to exit the program.

## Notes
- The script is designed to work with a single image (target-image.jpg). Adjust the filename and path as needed for other images.
- Ensure the img directory exists and contains the target image.