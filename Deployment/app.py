import os
from flask import Flask, render_template, request, send_from_directory
from keras.utils import load_img, img_to_array
import tensorflow as tf
import numpy as np
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './img/'

model = tf.keras.models.load_model('D:\Skripsi\Deployment\soybean_diseases_v12.h5')

target_names = ['Bacterial Blight', 'Cercospora Leaf Blight','Downy Mildew', 'Frogeye Leaf Spot',
'Healthy', 'Potassium Deficiency','Rust', 'Target Spot']

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/detect', methods=['GET'])
def disease_detection():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def predict():
    imagefile = request.files['imagefile'] 
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], imagefile.filename)

    imagefile.save(image_path)
    print(load_img(image_path).size)
    height, width = load_img(image_path).size
    
    convert = False
    image = load_img(image_path, target_size=(256,256))
    convert = True

    image1 = img_to_array(image)
    image1 = np.expand_dims(image1, axis=0)
    detection = False
    times = time.time()
    prediction = model.predict(image1)
    time_process = time.time() - times
    detection = True
    
    score = np.max(prediction)
    confidence=(round(100 * (score), 2))
    print(score)
    classes = np.argmax(prediction)
    if classes==0:
      result = target_names[classes]
      return render_template('bacterial_blight.html', result=result, confidence=f"{confidence}%", convert = convert,
                             height = height, width = width, detection = detection, time_process = round(time_process, 2), 
                             )
    elif classes==1:
      result = target_names[classes]
      return render_template('cercospora.html', result=result, confidence=f"{confidence}%", convert = convert,
                             height = height, width = width, detection = detection, time_process = round(time_process, 2), 
                             )
    elif classes==2:
      result = target_names[classes]
      return render_template('downy_mildew.html', result=result, confidence=f"{confidence}%", convert = convert,
                             height = height, width = width, detection = detection, time_process = round(time_process, 2), 
                             )
    elif classes==3:
      result = target_names[classes]
      return render_template('frogeye.html', result=result, confidence=f"{confidence}%", convert = convert,
                             height = height, width = width, detection = detection, time_process = round(time_process, 2), 
                             )
    elif classes==4:
      result = target_names[classes]
      return render_template('healthy.html', result=result, confidence=f"{confidence}%", convert = convert,
                             height = height, width = width, detection = detection, time_process = round(time_process, 2), 
                             )
    elif classes==5:
      result = target_names[classes]
      return render_template('potassium_deficiency.html', result=result, confidence=f"{confidence}%", convert = convert,
                             height = height, width = width, detection = detection, time_process = round(time_process, 2), 
                             )
    elif classes==6:
      result = target_names[classes]
      return render_template('rust.html', result=result, confidence=f"{confidence}%", convert = convert,
                             height = height, width = width, detection = detection, time_process = round(time_process, 2), 
                             )
    elif classes==7:
      result = target_names[classes]
      return render_template('target_spot.html', result=result, confidence=f"{confidence}%", convert = convert,
                             height = height, width = width, detection = detection, time_process = round(time_process, 2), 
                             )

@app.route('/img/<fileimg>')
def send_uploaded_image(fileimg=''):
    return send_from_directory(app.config['UPLOAD_FOLDER'], fileimg)

if __name__ == '__main__':
    app.run(port=3000, debug=True)