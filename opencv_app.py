import cv2 as cv # image processing library
import numpy as np # matrix math library
import streamlit as st # for the web app
from PIL import Image # for the image processing

# Defining the function to apply the image processing operations
def convert_to_gray(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return gray

# function to detect the edges
def detect_edges(img):
    edges = cv.Canny(img, 100, 200)
    return edges

# function to detect faces
def detect_faces(img):
    # original image face detection
    face_cascade=cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(img, 1.1, 4)
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return img
# set the title for web app
st.title("OpenCV Image Processing App")

# Add a button to upload the image file from user
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Add a button to perform the image processing operations
if uploaded_file is not None:
    # convert the file to an opencv image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv.imdecode(file_bytes, 1)
    # display the uploaded image
    st.image(img, channels="BGR",use_column_width=True)
    # when the button is clicked, apply the image processing operations
    if st.button("Convert to Gray"):
        gray = convert_to_gray(img)
        st.image(gray, use_column_width=True)
    if st.button("Detect Edges"):
        edges = detect_edges(img)
        st.image(edges, use_column_width=True)
    if st.button("Detect Faces"):
        faces = detect_faces(img)
        st.image(faces, use_column_width=True)
