from flask import Flask, request, jsonify
from flask_cors import CORS
from core.pipeline import run_pipeline

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ===== API =====
@app.route('/predict', methods=['POST'])
def predict_video():
    result, status = run_pipeline(files=request.files, form=request.form)
    return jsonify(result), status

    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
