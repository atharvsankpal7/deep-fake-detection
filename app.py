from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tensorflow as tf  # Use TensorFlow instead of PyTorch
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
MODEL_PATH = 'deepfake_detector.h5'  # Your actual model file

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Load deepfake detection model using TensorFlow
model = None
try:
    model = tf.keras.models.load_model(MODEL_PATH)  # Load Keras model
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({"success": False, "error": "No file part"}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"success": False, "error": "Invalid file format"}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        print(f"✅ File uploaded successfully: {filepath}")
        return jsonify({"success": True, "filepath": filepath}), 200
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze_video():
    try:
        uploaded_files = os.listdir(UPLOAD_FOLDER)
        if not uploaded_files:
            return jsonify({"success": False, "error": "No file uploaded for analysis"}), 400
        
        video_path = os.path.join(UPLOAD_FOLDER, uploaded_files[-1])  # Use latest uploaded file
        print(f"🔍 Analyzing video: {video_path}")

        if model is None:
            return jsonify({"success": False, "error": "Model not loaded"}), 500

        # TODO: Convert video to frames and preprocess for prediction
        # frames = process_video(video_path)  # Implement this function
        # predictions = model.predict(frames)  # Predict using Keras model
        # result = "Real" if predictions[0] > 0.5 else "Fake"  # Example classification

        result = "Fake"  # Temporary placeholder for now

        return jsonify({"success": True, "result": result}), 200

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
