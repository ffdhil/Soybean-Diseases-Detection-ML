import os
from flask import Flask, render_template, request, send_from_directory
from keras.utils import load_img, img_to_array
import tensorflow as tf
import numpy as np

app = Flask(__name__)
app.config['Upload_Folder'] = './'

model = tf.keras.models.load_model('D:\Deployment\soybean_diseases.h5')

target_names = ['Bacterial Blight', 'Cercospora Leaf Blight','Downy Mildew', 'Frogeye Leaf Spot',
'Healthy', 'Potassium Deficiency','Rust', 'Target Spot']

@app.route('/', methods=['GET'])
def disease_detection():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = os.path.join(app.config['Upload_Folder'], imagefile.filename)
    imagefile.save(image_path)
    image = load_img(image_path, target_size=(256,256))
    image1 = img_to_array(image)
    image1 = np.expand_dims(image1, axis=0)
    prediction = model.predict(image1)
    score = np.max(prediction)
    confidence=(round(100 * (score), 2))
    print(score)
    classes = np.argmax(prediction)
    if classes==0:
      result = target_names[classes]
      return render_template('bacterial_blight.html', result=result, confidence=f"{confidence}%")
    elif classes==1:
      result = target_names[classes]
      return render_template('cercospora.html', result=result, confidence=f"{confidence}%")
    elif classes==2:
      result = target_names[classes]
      return render_template('downy_mildew.html', result=result, confidence=f"{confidence}%")
    elif classes==3:
      result = target_names[classes]
      return render_template('frogeye.html', result=result, confidence=f"{confidence}%")
    elif classes==4:
      result = target_names[classes]
      return render_template('healthy.html', result=result, confidence=f"{confidence}%")
    elif classes==5:
      result = target_names[classes]
      return render_template('potassium_deficiency.html', result=result, confidence=f"{confidence}%")
    elif classes==6:
      result = target_names[classes]
      return render_template('rust.html', result=result, confidence=f"{confidence}%")
    elif classes==7:
      result = target_names[classes]
      return render_template('target_spot.html', result=result, confidence=f"{confidence}%")


@app.route('/img/<fileimg>')
def send_uploaded_image(fileimg=''):
    return send_from_directory(app.config['Upload_Folder'], fileimg)

if __name__ == '__main__':
    app.run(port=3000, debug=True)