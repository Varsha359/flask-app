from PIL import Image
import os

from flask import Flask, request, jsonify
from textDetection import detect_text
from new import main
import requests

app = Flask(__name__)
print("hello")


@app.route('/')  # decorator drfines the
def home():
    return "hello, this is our first flask website"


@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']
    # Replace this with the actual path to save the file
    file.save('../data/uploaded.jpeg')
    print("saved file")
    # Process the file here

    return 'Image uploaded successfully'


@app.route('/detect-text', methods=['GET'])
def detect_text_endpoint():
    # execute the Python script using subprocess
    # process = subprocess.Popen(
    #     ['python', 'src/line_segmented_image.py'], stdout=subprocess.PIPE)
    # output, error = process.communicate()
    print("called")
    detected_text_output = detect_text()
    print("segmented")
    # return the output of the script
    return detected_text_output


@app.route('/predict-medicines', methods=['GET'])
def predict_medicines_endpoint():
    response = requests.get('https://pharmafind.onrender.com/detect-text')
    # Check if the request was successful
    if response.status_code == 200:
        result = []
        # Get a list of all files in the directory
        print("called recognize")
        image_dir = '../output/uploaded_crops'
        file_list = os.listdir(image_dir)
        # Filter the list to include only image files
        image_files = [os.path.join(image_dir, f)
                       for f in file_list if f.endswith(('.png'))]
        i = 0
        for file_path in image_files:
            # Load the image from file
            # img = Image.open(file_path)
            recognized_text_output = main(file_path)
            print("recognized")
            result.append(recognized_text_output)
            i += 1
        # return the output of the script
        prediction_list = result
        return jsonify(prediction_list)
    else:
        # Return an error message if the request failed
        return jsonify({'error': 'Failed to get detect medicines'}), response.status_code


@app.route('/recognize-text', methods=['GET'])
def recognize_text_endpoint():
    result = []
    # Get a list of all files in the directory
    print("called recognize")
    image_dir = '../output/uploaded_crops'
    file_list = os.listdir(image_dir)
    # Filter the list to include only image files
    image_files = [os.path.join(image_dir, f)
                   for f in file_list if f.endswith(('.png'))]
    for file_path in image_files:
        # Load the image from file
        # img = Image.open(file_path)
        recognized_text_output = main(file_path)
        print("recognized")
        result.append(recognized_text_output)
    result.append("hi")
    # return the output of the script
    return {'output': result}
