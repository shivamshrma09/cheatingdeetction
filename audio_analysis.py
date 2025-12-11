import librosa
import numpy as np

class AudioAnalysisPipeline:
    def __init__(self):
        self.target_sr = 22050
        self.n_mfcc = 13
    
    def preprocess_audio(self, audio_path):
        """Load and preprocess audio"""
        try:
            audio, sr = librosa.load(audio_path, sr=self.target_sr)
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            return audio, sr
        except Exception as e:
            return None, None
    
    def detect_voice_activity(self, audio, sr, top_db=30):
        """Detect speech segments and pauses"""
        intervals = librosa.effects.split(audio, top_db=top_db)
        
        if len(intervals) == 0:
            return {
                'speech_duration': 0,
                'pause_duration': len(audio) / sr,
                'num_segments': 0,
                'avg_segment_duration': 0
            }
        
        speech_durations = [(end - start) / sr for start, end in intervals]
        total_speech = sum(speech_durations)
        total_audio = len(audio) / sr
        
        return {
            'speech_duration': total_speech,
            'pause_duration': total_audio - total_speech,
            'num_segments': len(intervals),
            'avg_segment_duration': np.mean(speech_durations),
            'speech_intervals': intervals
        }
    
    def analyze_pitch_variation(self, audio, sr, intervals):
        """Analyze pitch variation across speech segments"""
        if len(intervals) == 0:
            return {'mean_pitch': 0, 'pitch_std': 0, 'pitch_range': 0}
        
        pitches = []
        for start, end in intervals:
            segment = audio[start:end]
            if len(segment) > 0:
                f0, _ = librosa.pyin(segment, sr=sr, fmin=80, fmax=400)
                valid_f0 = f0[~np.isnan(f0)]
                if len(valid_f0) > 0:
                    pitches.extend(valid_f0)
        
        if len(pitches) == 0:
            return {'mean_pitch': 0, 'pitch_std': 0, 'pitch_range': 0}
        
        return {
            'mean_pitch': np.mean(pitches),
            'pitch_std': np.std(pitches),
            'pitch_range': np.max(pitches) - np.min(pitches)
        }
    
    def detect_background_noise(self, audio, sr, intervals):
        """Analyze background noise levels"""
        # Create mask for non-speech regions
        non_speech_mask = np.ones(len(audio), dtype=bool)
        for start, end in intervals:
            non_speech_mask[start:end] = False
        
        noise_audio = audio[non_speech_mask]
        if len(noise_audio) == 0:
            return {'noise_level': 0, 'noise_variability': 0}
        
        noise_rms = librosa.feature.rms(y=noise_audio)[0]
        return {
            'noise_level': np.mean(noise_rms),
            'noise_variability': np.std(noise_rms)
        }
    
    def detect_multiple_speakers(self, pitch_results, vad_results):
        """Simple multi-speaker detection based on pitch variation"""
        score = 0
        reasoning = []
        
        # High pitch variation suggests multiple speakers
        if pitch_results['pitch_range'] > 150:
            score += 4
            reasoning.append(f"High pitch range: {pitch_results['pitch_range']:.1f} Hz")
        
        if pitch_results['pitch_std'] > 40:
            score += 3
            reasoning.append(f"High pitch variability: {pitch_results['pitch_std']:.1f} Hz")
        
        # Many short segments suggest conversation
        if vad_results['num_segments'] > 5 and vad_results['avg_segment_duration'] < 2:
            score += 2
            reasoning.append(f"Many short segments: {vad_results['num_segments']} segments")
        
        return {
            'likelihood_score': score,
            'reasoning': '; '.join(reasoning) if reasoning else 'No strong indicators'
        }
    
    def calculate_fraud_score(self, vad_results, pitch_results, noise_results, speaker_results):
        """Calculate overall fraud score based on audio analysis"""
        fraud_score = 0
        
        # Suspicious pause patterns
        if vad_results['pause_duration'] > vad_results['speech_duration'] * 0.5:
            fraud_score += 2
        
        # Extreme pitch variations (could indicate voice modulation)
        if pitch_results['pitch_std'] > 50:
            fraud_score += 3
        
        # High background noise
        if noise_results['noise_level'] > 0.01:
            fraud_score += 1
        
        # Multiple speakers
        if speaker_results['likelihood_score'] >= 5:
            fraud_score += 4
        
        return min(fraud_score, 10)  # Cap at 10
    
    def analyze(self, audio_path):
        """Main audio analysis pipeline"""
        results = {
            'status': 'completed',
            'fraud_score': 0,
            'analysis_details': {}
        }
        
        # Preprocess audio
        audio, sr = self.preprocess_audio(audio_path)
        if audio is None:
            results['status'] = 'error'
            results['message'] = 'Failed to load audio'
            return results
        
        # Voice activity detection
        vad_results = self.detect_voice_activity(audio, sr)
        results['analysis_details']['voice_activity'] = vad_results
        
        # Pitch analysis
        intervals = vad_results.get('speech_intervals', [])
        pitch_results = self.analyze_pitch_variation(audio, sr, intervals)
        results['analysis_details']['pitch_analysis'] = pitch_results
        
        # Background noise analysis
        noise_results = self.detect_background_noise(audio, sr, intervals)
        results['analysis_details']['background_noise'] = noise_results
        
        # Multi-speaker detection
        speaker_results = self.detect_multiple_speakers(pitch_results, vad_results)
        results['analysis_details']['speaker_analysis'] = speaker_results
        
        # Calculate fraud score
        results['fraud_score'] = self.calculate_fraud_score(
            vad_results, pitch_results, noise_results, speaker_results
        )
        
        return results