from flask import Flask, render_template, request
import cv2
import numpy as np
import os

app = Flask(__name__)

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def binarize(image):
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return binary

def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations=1)

def opening(image):
    return dilate(erode(image))

def closing(image):
    return erode(dilate(image))

def count_objects(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No selected file')
    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)  # Save the uploaded file to a folder
        image = cv2.imread(filepath)
        grayscale_image = grayscale(image)
        binary_image = binarize(grayscale_image)
        return render_template('result.html', grayscale=grayscale_image, binary=binary_image)
    return render_template('index.html', error='Error in uploading file')


@app.route('/morphology', methods=['POST'])
def morphology():
    operation = request.form['operation']
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No selected file')
    if file:
        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        if operation == 'dilate':
            result = dilate(image)
        elif operation == 'erode':
            result = erode(image)
        elif operation == 'opening':
            result = opening(image)
        elif operation == 'closing':
            result = closing(image)
        return render_template('morphology_result.html', result=result)
    return render_template('index.html', error='Error in uploading file')

@app.route('/count_objects', methods=['POST'])
def count():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No selected file')
    if file:
        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        objects_count = count_objects(image)
        return render_template('count_result.html', count=objects_count)
    return render_template('index.html', error='Error in uploading file')

if __name__ == '__main__':
    app.run(debug=True)
