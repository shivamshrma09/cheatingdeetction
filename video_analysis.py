import cv2
import numpy as np
from deepface import DeepFace
import mediapipe as mp

class VideoAnalysisPipeline:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2, min_detection_confidence=0.5)
    
    def verify_face_match(self, video_path, reference_image_path):
        """Verify if person in video matches reference image"""
        if not reference_image_path:
            return {'verified': False, 'confidence': 0, 'message': 'No reference image provided'}
        
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        matches = 0
        confidence_scores = []
        
        while frame_count < 15:  # Check first 15 frames
            ret, frame = cap.read()
            if not ret:
                break
                
            try:
                result = DeepFace.verify(
                    img1_path=frame,
                    img2_path=reference_image_path,
                    model_name="VGG-Face",
                    distance_metric="cosine"
                )
                confidence_scores.append(result['distance'])
                if result['verified']:
                    matches += 1
            except:
                pass
            
            frame_count += 1
        
        cap.release()
        
        # Calculate overall confidence metrics
        match_confidence = matches / frame_count if frame_count > 0 else 0
        avg_distance = np.mean(confidence_scores) if confidence_scores else 1.0
        overall_confidence = (match_confidence * 0.7) + ((1 - avg_distance) * 0.3)
        
        # Confidence level assessment
        if overall_confidence >= 0.8:
            confidence_level = "Very High"
        elif overall_confidence >= 0.6:
            confidence_level = "High"
        elif overall_confidence >= 0.4:
            confidence_level = "Medium"
        elif overall_confidence >= 0.2:
            confidence_level = "Low"
        else:
            confidence_level = "Very Low"
        
        return {
            'verified': overall_confidence > 0.5,
            'confidence': round(overall_confidence, 3),
            'confidence_level': confidence_level,
            'matches': matches,
            'total_frames_checked': frame_count,
            'avg_distance': round(avg_distance, 3) if confidence_scores else None
        }
    
    def analyze_gaze_and_behavior(self, video_path):
        """Analyze eye gaze and suspicious behavior using MediaPipe"""
        cap = cv2.VideoCapture(video_path)
        
        # Tracking variables
        cheating_score = 0
        extreme_gaze_frames = 0
        multiple_face_frames = 0
        total_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            total_frames += 1
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Face detection
            detection_results = self.face_detection.process(rgb_frame)
            
            # Multiple faces detection
            if detection_results.detections and len(detection_results.detections) > 1:
                multiple_face_frames += 1
                cheating_score += 1
            
            # Face mesh for gaze analysis
            mesh_results = self.face_mesh.process(rgb_frame)
            
            if mesh_results.multi_face_landmarks:
                for face_landmarks in mesh_results.multi_face_landmarks:
                    # Simple gaze estimation using eye landmarks
                    # Left eye: landmarks 33, 133, 160, 158, 144, 153
                    # Right eye: landmarks 362, 398, 384, 385, 386, 387
                    
                    h, w = frame.shape[:2]
                    
                    # Get eye center points
                    left_eye_center = face_landmarks.landmark[33]
                    right_eye_center = face_landmarks.landmark[362]
                    nose_tip = face_landmarks.landmark[1]
                    
                    # Convert to pixel coordinates
                    left_eye_x = int(left_eye_center.x * w)
                    right_eye_x = int(right_eye_center.x * w)
                    nose_x = int(nose_tip.x * w)
                    
                    # Simple gaze direction check
                    eye_center_x = (left_eye_x + right_eye_x) // 2
                    
                    # If eyes are looking significantly away from nose center
                    if abs(eye_center_x - nose_x) > 30:
                        extreme_gaze_frames += 1
                        if extreme_gaze_frames % 30 == 0:  # Every 30 frames of looking away
                            cheating_score += 1
        
        cap.release()
        
        return {
            'total_frames': total_frames,
            'extreme_gaze_frames': extreme_gaze_frames,
            'multiple_face_frames': multiple_face_frames,
            'cheating_score': cheating_score,
            'suspicious_behavior_percentage': (extreme_gaze_frames + multiple_face_frames) / total_frames * 100 if total_frames > 0 else 0
        }
    
    def analyze_head_pose(self, video_path):
        """Analyze head pose for suspicious movements"""
        cap = cv2.VideoCapture(video_path)
        extreme_movements = 0
        total_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            total_frames += 1
            
            try:
                # Use DeepFace for pose analysis
                analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                if analysis and len(analysis) > 0 and 'pose' in analysis[0]:
                    pose = analysis[0]['pose']
                    yaw, pitch, roll = pose['yaw'], pose['pitch'], pose['roll']
                    
                    # Check for extreme movements
                    if abs(yaw) > 30 or abs(pitch) > 20 or abs(roll) > 15:
                        extreme_movements += 1
            except:
                pass
        
        cap.release()
        
        return {
            'total_frames': total_frames,
            'extreme_movement_frames': extreme_movements,
            'extreme_movement_percentage': extreme_movements / total_frames * 100 if total_frames > 0 else 0
        }
    
    def analyze_blink_patterns(self, video_path):
        """Analyze eye blink patterns for suspicious behavior"""
        cap = cv2.VideoCapture(video_path)
        blink_count = 0
        rapid_blinks = 0
        no_blink_frames = 0
        total_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            total_frames += 1
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mesh_results = self.face_mesh.process(rgb_frame)
            
            if mesh_results.multi_face_landmarks:
                for face_landmarks in mesh_results.multi_face_landmarks:
                    # Eye aspect ratio calculation
                    left_eye = [face_landmarks.landmark[i] for i in [33, 7, 163, 144, 145, 153]]
                    right_eye = [face_landmarks.landmark[i] for i in [362, 382, 381, 380, 374, 373]]
                    
                    # Simple blink detection based on eye height
                    left_ear = abs(left_eye[1].y - left_eye[5].y) / abs(left_eye[0].x - left_eye[3].x)
                    right_ear = abs(right_eye[1].y - right_eye[5].y) / abs(right_eye[0].x - right_eye[3].x)
                    ear = (left_ear + right_ear) / 2.0
                    
                    if ear < 0.2:  # Blink detected
                        blink_count += 1
                    if ear < 0.15:  # Very closed eyes
                        no_blink_frames += 1
        
        cap.release()
        blink_rate = blink_count / (total_frames / 30) if total_frames > 0 else 0  # per second
        
        return {
            'total_blinks': blink_count,
            'blink_rate_per_second': round(blink_rate, 2),
            'suspicious_blink_pattern': blink_rate > 2 or blink_rate < 0.1,
            'no_blink_percentage': (no_blink_frames / total_frames * 100) if total_frames > 0 else 0
        }
    
    def analyze_emotions(self, video_path):
        """Analyze emotions for stress/deception indicators"""
        cap = cv2.VideoCapture(video_path)
        emotions_detected = []
        stress_indicators = 0
        total_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            total_frames += 1
            if total_frames % 30 == 0:  # Analyze every 30th frame
                try:
                    analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                    if analysis and len(analysis) > 0:
                        emotion = analysis[0]['dominant_emotion']
                        emotions_detected.append(emotion)
                        
                        # Stress indicators
                        if emotion in ['angry', 'fear', 'sad']:
                            stress_indicators += 1
                except:
                    pass
        
        cap.release()
        
        return {
            'dominant_emotions': list(set(emotions_detected)),
            'stress_indicator_count': stress_indicators,
            'stress_percentage': (stress_indicators / len(emotions_detected) * 100) if emotions_detected else 0,
            'emotional_stability': len(set(emotions_detected)) <= 2
        }
    
    def analyze_face_distance(self, video_path):
        """Analyze face distance variations"""
        cap = cv2.VideoCapture(video_path)
        face_sizes = []
        total_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            total_frames += 1
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detection_results = self.face_detection.process(rgb_frame)
            
            if detection_results.detections:
                for detection in detection_results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    face_area = bbox.width * bbox.height
                    face_sizes.append(face_area)
        
        cap.release()
        
        if face_sizes:
            avg_size = np.mean(face_sizes)
            size_variation = np.std(face_sizes)
            return {
                'avg_face_size': round(avg_size, 4),
                'size_variation': round(size_variation, 4),
                'suspicious_movement': size_variation > 0.02,
                'distance_changes': len([s for s in face_sizes if abs(s - avg_size) > avg_size * 0.3])
            }
        
        return {'error': 'No face detected for distance analysis'}
    
    def detect_mobile_phone(self, video_path):
        """Simple mobile phone detection using color and shape analysis"""
        cap = cv2.VideoCapture(video_path)
        phone_detected_frames = 0
        total_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            total_frames += 1
            
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Define range for black/dark colors (typical phone colors)
            lower_dark = np.array([0, 0, 0])
            upper_dark = np.array([180, 255, 50])
            
            # Create mask and find contours
            mask = cv2.inRange(hsv, lower_dark, upper_dark)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Look for rectangular shapes (potential phones)
            for contour in contours:
                area = cv2.contourArea(contour)
                if 1000 < area < 50000:  # Phone-like size
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    if 0.4 < aspect_ratio < 2.5:  # Phone-like aspect ratio
                        phone_detected_frames += 1
                        break
        
        cap.release()
        
        return {
            'phone_detected_frames': phone_detected_frames,
            'phone_detection_percentage': (phone_detected_frames / total_frames * 100) if total_frames > 0 else 0,
            'suspicious_object_present': phone_detected_frames > total_frames * 0.1
        }
    
    def analyze_background_changes(self, video_path):
        """Detect background movements and changes"""
        cap = cv2.VideoCapture(video_path)
        ret, prev_frame = cap.read()
        if not ret:
            return {'error': 'Could not read video'}
        
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        background_changes = 0
        total_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            total_frames += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate frame difference
            diff = cv2.absdiff(prev_gray, gray)
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            
            # Count non-zero pixels (changes)
            change_pixels = cv2.countNonZero(thresh)
            total_pixels = thresh.shape[0] * thresh.shape[1]
            change_percentage = change_pixels / total_pixels
            
            if change_percentage > 0.1:  # Significant background change
                background_changes += 1
            
            prev_gray = gray
        
        cap.release()
        
        return {
            'background_change_frames': background_changes,
            'background_stability': (1 - background_changes / total_frames) if total_frames > 0 else 1,
            'suspicious_background_activity': background_changes > total_frames * 0.2
        }
    
    def analyze(self, video_path, reference_image_path=None):
        """Enhanced analysis pipeline with all features"""
        results = {
            'status': 'completed',
            'fraud_score': 0,
            'analysis_details': {}
        }
        
        # Face verification
        if reference_image_path:
            face_match = self.verify_face_match(video_path, reference_image_path)
            results['analysis_details']['face_verification'] = face_match
            if not face_match['verified']:
                results['fraud_score'] += 5
        
        # Gaze and behavior analysis
        gaze_analysis = self.analyze_gaze_and_behavior(video_path)
        results['analysis_details']['gaze_behavior'] = gaze_analysis
        results['fraud_score'] += min(gaze_analysis['cheating_score'], 5)
        
        # Head pose analysis
        pose_analysis = self.analyze_head_pose(video_path)
        results['analysis_details']['head_pose'] = pose_analysis
        if pose_analysis['extreme_movement_percentage'] > 20:
            results['fraud_score'] += 3
        
        # Blink pattern analysis
        blink_analysis = self.analyze_blink_patterns(video_path)
        results['analysis_details']['blink_patterns'] = blink_analysis
        if blink_analysis['suspicious_blink_pattern']:
            results['fraud_score'] += 2
        
        # Emotion analysis
        emotion_analysis = self.analyze_emotions(video_path)
        results['analysis_details']['emotions'] = emotion_analysis
        if emotion_analysis['stress_percentage'] > 50:
            results['fraud_score'] += 2
        
        # Face distance analysis
        distance_analysis = self.analyze_face_distance(video_path)
        results['analysis_details']['face_distance'] = distance_analysis
        if distance_analysis.get('suspicious_movement', False):
            results['fraud_score'] += 2
        
        # Mobile phone detection
        phone_analysis = self.detect_mobile_phone(video_path)
        results['analysis_details']['mobile_detection'] = phone_analysis
        if phone_analysis['suspicious_object_present']:
            results['fraud_score'] += 4
        
        # Background analysis
        background_analysis = self.analyze_background_changes(video_path)
        results['analysis_details']['background_analysis'] = background_analysis
        if background_analysis['suspicious_background_activity']:
            results['fraud_score'] += 3
        
        return results