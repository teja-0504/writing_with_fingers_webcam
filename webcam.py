"""
Webcam and Hand Tracking Module for Server-Side Processing
This module handles configuration, drawing data processing, and file management.
"""

import json
import uuid
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class WebcamConfig:
    """Configuration management for the webcam application"""
    
    def __init__(self):
        self.colors = [
            {"name": "Red", "value": "#FF0000"},
            {"name": "Blue", "value": "#0000FF"},
            {"name": "Green", "value": "#00FF00"},
            {"name": "Purple", "value": "#800080"},
            {"name": "Orange", "value": "#FFA500"},
            {"name": "Gold", "value": "#FFD700"},
            {"name": "Pink", "value": "#FF69B4"},
            {"name": "Black", "value": "#000000"}
        ]
        
        self.detection_settings = {
            "maxNumHands": 1,
            "modelComplexity": 1,
            "minDetectionConfidence": 0.5,
            "minTrackingConfidence": 0.5
        }
        
        self.gesture_modes = {
            1: {"name": "Move", "description": "Track finger position"},
            2: {"name": "Drawing", "description": "Draw with current color"},
            3: {"name": "Change Color", "description": "Hold 1 second to change color"},
            4: {"name": "Clear All", "description": "Clear entire canvas"},
            5: {"name": "Undo Last Line", "description": "Remove last drawn line"}
        }
    
    def get_config(self):
        """Get complete configuration"""
        return {
            "colors": self.colors,
            "detection_settings": self.detection_settings,
            "gesture_modes": self.gesture_modes
        }

class DrawingManager:
    """Manages drawing data and file operations"""
    
    def __init__(self, drawings_dir="drawings"):
        self.drawings_dir = drawings_dir
        os.makedirs(self.drawings_dir, exist_ok=True)
    
    def save_drawing(self, drawing_data):
        """Save drawing data to file"""
        try:
            if not drawing_data:
                raise ValueError("No drawing data provided")
            
            # Generate unique drawing ID
            drawing_id = f"drawing_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
            
            # Add metadata
            drawing_data['metadata'] = {
                'id': drawing_id,
                'created_at': datetime.now().isoformat(),
                'version': '1.0',
                'canvas_size': drawing_data.get('canvas_size', {'width': 640, 'height': 480}),
                'total_lines': len(drawing_data.get('lines', [])),
                'colors_used': len(set(
                    segment.get('color', '#000000') 
                    for line in drawing_data.get('lines', []) 
                    for segment in line
                ))
            }
            
            # Save to file
            filename = f"{drawing_id}.json"
            filepath = os.path.join(self.drawings_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(drawing_data, f, indent=2)
            
            logger.info(f"Drawing saved: {drawing_id}")
            return {
                "status": "success",
                "drawing_id": drawing_id,
                "message": "Drawing saved successfully",
                "filepath": filepath
            }
            
        except Exception as e:
            logger.error(f"Error saving drawing: {e}")
            return {
                "status": "error",
                "message": f"Failed to save drawing: {str(e)}"
            }
    
    def list_drawings(self):
        """List all saved drawings"""
        try:
            drawings = []
            for filename in os.listdir(self.drawings_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.drawings_dir, filename)
                    try:
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                            metadata = data.get('metadata', {})
                            drawings.append({
                                'id': metadata.get('id', filename),
                                'created_at': metadata.get('created_at'),
                                'filename': filename,
                                'canvas_size': metadata.get('canvas_size'),
                                'total_lines': metadata.get('total_lines', 0),
                                'colors_used': metadata.get('colors_used', 0)
                            })
                    except Exception as e:
                        logger.warning(f"Error reading drawing file {filename}: {e}")
            
            # Sort by creation date (newest first)
            drawings.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            return {"drawings": drawings}
            
        except Exception as e:
            logger.error(f"Error listing drawings: {e}")
            return {"error": f"Failed to list drawings: {str(e)}"}
    
    def get_drawing(self, drawing_id):
        """Get specific drawing by ID"""
        try:
            for filename in os.listdir(self.drawings_dir):
                if filename.startswith(f"drawing_") and drawing_id in filename:
                    filepath = os.path.join(self.drawings_dir, filename)
                    with open(filepath, 'r') as f:
                        return json.load(f)
            
            return {"error": "Drawing not found"}
            
        except Exception as e:
            logger.error(f"Error getting drawing {drawing_id}: {e}")
            return {"error": f"Failed to get drawing: {str(e)}"}
    
    def delete_drawing(self, drawing_id):
        """Delete specific drawing by ID"""
        try:
            for filename in os.listdir(self.drawings_dir):
                if filename.startswith(f"drawing_") and drawing_id in filename:
                    filepath = os.path.join(self.drawings_dir, filename)
                    os.remove(filepath)
                    logger.info(f"Drawing deleted: {drawing_id}")
                    return {
                        "status": "success",
                        "message": f"Drawing {drawing_id} deleted successfully"
                    }
            
            return {"error": "Drawing not found"}
            
        except Exception as e:
            logger.error(f"Error deleting drawing {drawing_id}: {e}")
            return {"error": f"Failed to delete drawing: {str(e)}"}

class GestureProcessor:
    """Process gesture data and validate finger counting logic"""
    
    @staticmethod
    def validate_finger_count(finger_count):
        """Validate finger count is within expected range"""
        return 0 <= finger_count <= 5
    
    @staticmethod
    def get_gesture_action(finger_count):
        """Get the action associated with a finger count"""
        actions = {
            1: "move",
            2: "draw", 
            3: "change_color",
            4: "clear_all",
            5: "undo_last"
        }
        return actions.get(finger_count, "unknown")
    
    @staticmethod
    def process_drawing_data(lines_data):
        """Process and validate drawing data"""
        if not isinstance(lines_data, list):
            return {"error": "Lines data must be a list"}
        
        processed_lines = []
        total_segments = 0
        
        for line in lines_data:
            if not isinstance(line, list):
                continue
                
            processed_line = []
            for segment in line:
                if isinstance(segment, dict) and all(key in segment for key in ['from', 'to', 'color', 'size']):
                    processed_line.append(segment)
                    total_segments += 1
            
            if processed_line:
                processed_lines.append(processed_line)
        
        return {
            "lines": processed_lines,
            "total_lines": len(processed_lines),
            "total_segments": total_segments
        }

# Global instances
config_manager = WebcamConfig()
drawing_manager = DrawingManager()
gesture_processor = GestureProcessor()

# Flask Application
from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__)

@app.route('/')
def index():
    """Serve the main application page"""
    return render_template('index.html')

@app.route('/api/config')
def get_config():
    """Get application configuration"""
    try:
        return jsonify(config_manager.get_config())
    except Exception as e:
        logger.error(f"Error getting config: {e}")
        return jsonify({"error": "Failed to get configuration"}), 500

@app.route('/api/save-drawing', methods=['POST'])
def save_drawing():
    """Save drawing data to file"""
    try:
        drawing_data = request.get_json()
        
        if not drawing_data:
            return jsonify({"status": "error", "message": "No drawing data provided"}), 400
        
        # Process drawing data
        processed_data = gesture_processor.process_drawing_data(drawing_data.get('lines', []))
        if 'error' in processed_data:
            return jsonify({"status": "error", "message": processed_data['error']}), 400
        
        # Add processed data back to drawing
        drawing_data['lines'] = processed_data['lines']
        drawing_data['stats'] = {
            'total_lines': processed_data['total_lines'],
            'total_segments': processed_data['total_segments']
        }
        
        # Save drawing
        result = drawing_manager.save_drawing(drawing_data)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error saving drawing: {e}")
        return jsonify({
            "status": "error",
            "message": f"Failed to save drawing: {str(e)}"
        }), 500

@app.route('/api/drawings')
def list_drawings():
    """List all saved drawings"""
    try:
        result = drawing_manager.list_drawings()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error listing drawings: {e}")
        return jsonify({"error": "Failed to list drawings"}), 500

@app.route('/api/drawings/<drawing_id>')
def get_drawing(drawing_id):
    """Get specific drawing by ID"""
    try:
        result = drawing_manager.get_drawing(drawing_id)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error getting drawing: {e}")
        return jsonify({"error": "Failed to get drawing"}), 500

@app.route('/api/drawings/<drawing_id>', methods=['DELETE'])
def delete_drawing(drawing_id):
    """Delete specific drawing by ID"""
    try:
        result = drawing_manager.delete_drawing(drawing_id)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error deleting drawing: {e}")
        return jsonify({"error": "Failed to delete drawing"}), 500

@app.route('/api/validate-gesture', methods=['POST'])
def validate_gesture():
    """Validate gesture data"""
    try:
        gesture_data = request.get_json()
        finger_count = gesture_data.get('finger_count', 0)
        
        if not gesture_processor.validate_finger_count(finger_count):
            return jsonify({"valid": False, "message": "Invalid finger count"}), 400
        
        action = gesture_processor.get_gesture_action(finger_count)
        return jsonify({
            "valid": True,
            "finger_count": finger_count,
            "action": action,
            "message": f"Gesture recognized: {action}"
        })
        
    except Exception as e:
        logger.error(f"Error validating gesture: {e}")
        return jsonify({"error": "Failed to validate gesture"}), 500

@app.route('/shutdown', methods=['POST'])
def shutdown():
    """Shutdown the application"""
    try:
        logger.info("Shutdown requested")
        # In production, this should be handled differently
        # For now, just return success
        return jsonify({"status": "success", "message": "Shutdown initiated"})
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
        return jsonify({"error": "Shutdown failed"}), 500

@app.route('/health')
def health():
    """Health check endpoint for deployment"""
    return jsonify({
        "status": "healthy", 
        "message": "Finger painting app with client-side hand tracking is running",
        "mode": "client-side",
        "version": "1.0.0"
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
