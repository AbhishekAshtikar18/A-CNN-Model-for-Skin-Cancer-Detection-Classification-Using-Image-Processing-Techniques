from flask import Flask, render_template, request, flash, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_required, current_user, login_user, logout_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import cv2
import base64
from io import BytesIO


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:root@localhost/skinnewdb'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
app.config['UPLOAD_FOLDER'] = 'static/uploads'

db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(255), unique=True)
    email = db.Column(db.String(255), unique=True)
    password = db.Column(db.String(255))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template("index.html")

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user:
            flash('Email address already exists.')
            return redirect(url_for('register'))
        new_user = User(username=username, email=email, password=generate_password_hash(password, method='sha256'))
        db.session.add(new_user)
        db.session.commit()
        return render_template("login.html")
    else:
        return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if not user or not check_password_hash(user.password, password):
            flash('Invalid Username & Password')
            return render_template("login.html",aa="Invalid Username & Password")
        else:
            login_user(user)
            return redirect(url_for('predict'))
        
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

# Load the trained model for skin disease classification
model = tf.keras.models.load_model('skin.h5')

class_labels = [
    'Actinic Keratosis', 'Basal Cell Carcinoma', 'Dermatofibroma',
    'Healthy', 'Melanoma', 'Nevus', 'Pigmented Benign Keratosis',
    'Seborrheic Keratosis', 'Squamous Cell Carcinoma', 'Vascular Lesion'
]

class_benefits = {
    'Actinic Keratosis': {
        'cause': 'Caused by sun exposure and UV radiation',
        'treatment': 'Treatment includes cryotherapy, laser therapy, or topical medications'
    },
    'Basal Cell Carcinoma': {
        'cause': 'Common form of skin cancer, often caused by UV exposure',
        'treatment': 'Treatment includes surgery, radiation therapy, or topical medications'
    },
    'Dermatofibroma': {
        'cause': 'Benign skin growth often caused by minor trauma',
        'treatment': 'Typically benign and may not require treatment, but can be removed for cosmetic reasons'
    },
    'Healthy': {
        'cause': 'Healthy skin',
        'treatment': 'No specific treatment required; maintain good skin care'
    },
    'Melanoma': {
        'cause': 'Serious form of skin cancer, often caused by UV exposure',
        'treatment': 'Treatment includes surgery, chemotherapy, immunotherapy, or targeted therapy'
    },
    'Nevus': {
        'cause': 'Benign mole or birthmark',
        'treatment': 'Typically benign and may not require treatment, but should be monitored for changes'
    },
    'Pigmented Benign Keratosis': {
        'cause': 'Common skin lesion often associated with aging and sun exposure',
        'treatment': 'Treatment includes cryotherapy or removal for cosmetic reasons'
    },
    'Seborrheic Keratosis': {
        'cause': 'Benign skin growth often associated with aging',
        'treatment': 'Typically benign and may not require treatment, but can be removed for cosmetic reasons'
    },
    'Squamous Cell Carcinoma': {
        'cause': 'Skin cancer, often caused by UV exposure',
        'treatment': 'Treatment includes surgery, radiation therapy, or topical medications'
    },
    'Vascular Lesion': {
        'cause': 'Skin lesion involving blood vessels',
        'treatment': 'Treatment varies depending on the type of vascular lesion and may include laser therapy or surgical removal'
    },
    'Invalid input': {
        'cause': 'The uploaded image does not match any known skin condition in the dataset.',
        'treatment': 'Please upload a clear image of a skin condition.'
    }
}

def is_valid_skin_image(image):
    # Convert the image to RGB
    image_rgb = image.convert('RGB')
    image_np = np.array(image_rgb)

    # Convert RGB to HSV
    image_hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)

    # Define lower and upper bounds for skin color in HSV
    lower_skin = np.array([0, 40, 50], dtype=np.uint8)
    upper_skin = np.array([25, 255, 255], dtype=np.uint8)

    # Create a mask for skin color detection
    mask = cv2.inRange(image_hsv, lower_skin, upper_skin)

    # Calculate percentage of skin pixels
    skin_percentage = (np.sum(mask == 255) / mask.size) * 100

    # Threshold for skin detection
    if skin_percentage >= 5:  # Adjust this threshold based on your needs
        return True
    else:
        return False

@app.route('/prediction', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        # Get the image file from the request
        image_file = request.files['image']

        # Save the image file to the UPLOAD_FOLDER directory
        filename = secure_filename(image_file.filename)
        image_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Read the image using Pillow
        image = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Check if the image is a valid skin image
        if not is_valid_skin_image(image):
            pred_label = 'Invalid input'
            pred_cause = class_benefits['Invalid input']['cause']
            pred_treatment = class_benefits['Invalid input']['treatment']
            pred_probability = 'N/A'
        else:
            # Preprocess the image for the skin model
            image_resized = image.resize((224, 224))
            image_np = np.array(image_resized) / 255.0
            image_np = np.expand_dims(image_np, axis=0)

            # Perform prediction using your model
            pred_probs = model.predict(image_np)[0]

            # Get the predicted class label and probability
            max_prob = max(pred_probs)
            pred_label = class_labels[np.argmax(pred_probs)]
            pred_cause = class_benefits[pred_label]['cause']
            pred_treatment = class_benefits[pred_label]['treatment']
            pred_probability = f"{max_prob * 100:.2f}%"

        # Build the response dictionary
        prediction = {
            'label': pred_label,
            'cause': pred_cause,
            'treatment': pred_treatment,
            'probability': pred_probability
        }

        image_url = url_for('static', filename=f'uploads/{filename}')

        # Render the HTML template with the prediction result and image
        return render_template('prediction.html', prediction=prediction, image=image_url)

    # Render the HTML form to upload an image
    return render_template('prediction.html')

@app.route('/live', methods=['GET', 'POST'])
@login_required
def live():
    if request.method == 'POST':
        # Get the image data from the form
        image_data = request.form['image']

        # Decode the base64 image data
        image_data = image_data.split(',')[1]
        image = Image.open(BytesIO(base64.b64decode(image_data)))

        # Save the image to the upload folder (optional)
        filename = 'captured_image.png'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(filepath)

        # Preprocess the image
        image = image.resize((224, 224))
        image = np.array(image.convert('RGB'))  # Ensure RGB format
        image = image / 255.0
        image = np.expand_dims(image, axis=0)

        # Get prediction
        pred_probs = model.predict(image)[0]
        max_prob = max(pred_probs)
        pred_label = class_labels[np.argmax(pred_probs)]
        pred_cause = class_benefits[pred_label]['cause']
        pred_treatment = class_benefits[pred_label]['treatment']
        pred_probability = f"{max_prob * 100:.2f}%"
        # Prepare prediction data
        prediction = {
            'label': pred_label,
            'cause': pred_cause,
            'treatment': pred_treatment,
            'probability': pred_probability
        }

        image_url = url_for('static', filename='uploads/' + filename)
        return render_template('live.html', prediction=prediction, image=image_url)

    return render_template('live.html')


@app.route('/graphs', methods=['POST', 'GET'])
@login_required
def graphs():
    return render_template("graphs.html")

if __name__ == "__main__":
    app.run(debug=True)
