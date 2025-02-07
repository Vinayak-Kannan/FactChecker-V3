from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from ClusterAndPredict.ClusterAndPredict import ClusterAndPredict
from test_single_claim import test_single_claim_processing, load_s3_data
from dotenv import load_dotenv

load_dotenv()  # 默认加载当前工作目录下的 .env 文件


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Create and configure the Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

@app.route('/api/verify', methods=['POST'])
def verify_claim():
    """
    API endpoint to verify a claim.
    It expects a JSON body with a 'claim' key.
    """
    data = request.get_json()
    if not data or 'claim' not in data:
        return jsonify({"error": "Missing claim parameter"}), 400

    claim_text = data['claim'].strip()
    if not claim_text:
        return jsonify({"error": "Empty claim"}), 400

    try:
        # Process the single claim using the model
        result = test_single_claim_processing(claim_text)
        response = {
            "claim": result.get("claim", claim_text),
            "prediction": result.get("prediction", "Error"),
            "confidence": result.get("confidence", 0)
            # Optionally add other fields such as explanation, cluster, etc.
        }
        return jsonify(response)
    except Exception as e:
        logger.error(f"API error: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Run the Flask app in development mode (for production, use a WSGI server like Gunicorn)
    app.run(host='0.0.0.0', port=5000, debug=False)
