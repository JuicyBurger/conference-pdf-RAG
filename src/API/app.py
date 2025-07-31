"""
Sinon-RAG API Application

Main Flask application for real-time chat and PDF document processing.
"""

import asyncio
import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

# Add the API directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import route handlers
from routes.chat import chat_bp
from routes.upload import upload_bp
from routes.status import status_bp
from utils.response import error_response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_app():
    """Create and configure Flask application"""
    app = Flask(__name__)
    
    # Configuration
    app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max request size
    app.config['UPLOAD_FOLDER'] = 'data/uploads'
    
    # Enable CORS for all routes
    CORS(app, origins="*", supports_credentials=False)
    
    # Register blueprints
    app.register_blueprint(chat_bp, url_prefix='/api/chat')
    app.register_blueprint(upload_bp, url_prefix='/api/upload')
    app.register_blueprint(status_bp, url_prefix='/api/status')
    
    # Health check endpoint
    @app.route('/', methods=['GET'])
    @app.route('/health', methods=['GET'])
    def health_check():
        """API health check"""
        return jsonify({
            'status': 'healthy',
            'service': 'sinon-rag-api',
            'version': '1.0.0',
            'endpoints': [
                '/api/chat/message',
                '/api/chat/history/<room_id>',
                '/api/upload/pdf',
                '/api/upload/status/<task_id>',
                '/api/status/progress/<task_id>'
            ]
        })
    
    # Global error handlers
    @app.errorhandler(404)
    def not_found_error(error):
        return jsonify(error_response("Endpoint not found", "NOT_FOUND")), 404
    
    @app.errorhandler(400)
    def bad_request_error(error):
        return jsonify(error_response("Bad request", "BAD_REQUEST")), 400
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error: {error}")
        return jsonify(error_response("Internal server error", "INTERNAL_ERROR")), 500
    
    @app.errorhandler(413)
    def file_too_large_error(error):
        return jsonify(error_response(
            "File too large. Maximum size is 500MB",
            "FILE_TOO_LARGE"
        )), 413
    
    # Handle OPTIONS requests for CORS
    @app.before_request
    def handle_preflight():
        if request.method == "OPTIONS":
            return jsonify({}), 200
    
    return app

# Create app instance
app = create_app()

if __name__ == '__main__':
    # Ensure data directories exist
    os.makedirs('data/uploads', exist_ok=True)
    
    # Run the app
    logger.info("Starting Sinon-RAG API server...")
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    ) 