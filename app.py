from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
from werkzeug.utils import secure_filename
from video_analysis import VideoAnalysisPipeline
from audio_analysis import AudioAnalysisPipeline
from code_analysis import CodeAnalysisPipeline

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a'}
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp'}

def allowed_file(filename, extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions

@app.route('/interview-monitor', methods=['POST'])
def interview_monitor():
    """
    Interview time monitoring - Face matching + Mobile detection
    Accepts: video, reference_image
    Returns: Real-time fraud detection results
    """
    try:
        results = {
            'status': 'success',
            'fraud_score': 0,
            'risk_level': 'low',
            'face_verification': {},
            'mobile_detection': {},
            'recommendations': []
        }
        
        video_pipeline = VideoAnalysisPipeline()
        
        # Process video (required)
        if 'video' not in request.files:
            return jsonify({'status': 'error', 'message': 'Video file required'}), 400
            
        video_file = request.files['video']
        if not video_file or not allowed_file(video_file.filename, ALLOWED_VIDEO_EXTENSIONS):
            return jsonify({'status': 'error', 'message': 'Invalid video file'}), 400
            
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
            video_file.save(tmp_video.name)
            
            # Get reference image
            ref_image_path = None
            if 'reference_image' in request.files:
                ref_img = request.files['reference_image']
                if ref_img and allowed_file(ref_img.filename, ALLOWED_IMAGE_EXTENSIONS):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_img:
                        ref_img.save(tmp_img.name)
                        ref_image_path = tmp_img.name
            
            # Face verification with confidence assessment
            face_results = video_pipeline.verify_face_match(tmp_video.name, ref_image_path)
            results['face_verification'] = face_results
            
            # Adjust fraud score based on confidence level
            if not face_results.get('verified', False):
                results['fraud_score'] += 8
            elif face_results.get('confidence_level') == 'Low':
                results['fraud_score'] += 4
            elif face_results.get('confidence_level') == 'Very Low':
                results['fraud_score'] += 6
            
            # Mobile/multiple person detection
            behavior_results = video_pipeline.analyze_gaze_and_behavior(tmp_video.name)
            results['mobile_detection'] = {
                'multiple_faces_detected': behavior_results['multiple_face_frames'] > 0,
                'suspicious_behavior_score': behavior_results['cheating_score'],
                'details': behavior_results
            }
            results['fraud_score'] += min(behavior_results['cheating_score'], 7)
            
            # Cleanup
            os.unlink(tmp_video.name)
            if ref_image_path:
                os.unlink(ref_image_path)
        
        # Risk assessment
        if results['fraud_score'] >= 10:
            results['risk_level'] = 'high'
            results['recommendations'].append('Stop interview - High fraud risk detected')
        elif results['fraud_score'] >= 5:
            results['risk_level'] = 'medium'
            results['recommendations'].append('Monitor closely - Suspicious activity detected')
        else:
            results['risk_level'] = 'low'
            results['recommendations'].append('Continue interview - No major issues detected')
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/post-interview-analysis', methods=['POST'])
def post_interview_analysis():
    """
    Post-interview analysis - Audio + Code analysis
    Accepts: audio, code_text
    Returns: Comprehensive fraud analysis
    """
    try:
        results = {
            'status': 'success',
            'overall_fraud_score': 0,
            'risk_level': 'low',
            'audio_analysis': {},
            'code_analysis': {},
            'recommendations': []
        }
        
        audio_pipeline = AudioAnalysisPipeline()
        code_pipeline = CodeAnalysisPipeline()
        
        # Process audio if provided
        if 'audio' in request.files:
            audio_file = request.files['audio']
            if audio_file and allowed_file(audio_file.filename, ALLOWED_AUDIO_EXTENSIONS):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_audio:
                    audio_file.save(tmp_audio.name)
                    audio_results = audio_pipeline.analyze(tmp_audio.name)
                    results['audio_analysis'] = audio_results
                    results['overall_fraud_score'] += audio_results.get('fraud_score', 0)
                    os.unlink(tmp_audio.name)
        
        # Process code if provided
        code_text = request.form.get('code_text')
        if code_text:
            code_results = code_pipeline.analyze(code_text)
            results['code_analysis'] = code_results
            results['overall_fraud_score'] += code_results.get('fraud_score', 0)
        
        # Final assessment
        if results['overall_fraud_score'] >= 12:
            results['risk_level'] = 'critical'
            results['recommendations'].append('Reject submission - High fraud probability')
        elif results['overall_fraud_score'] >= 8:
            results['risk_level'] = 'high'
            results['recommendations'].append('Manual review required')
        elif results['overall_fraud_score'] >= 4:
            results['risk_level'] = 'medium'
            results['recommendations'].append('Additional verification needed')
        else:
            results['risk_level'] = 'low'
            results['recommendations'].append('Submission appears legitimate')
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'Anti-Cheating AI System'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(debug=False, host='0.0.0.0', port=port)
