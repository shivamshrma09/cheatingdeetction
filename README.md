---
title: Anti-Cheating AI System
emoji: 🔍
colorFrom: red
colorTo: orange
sdk: docker
app_file: app.py
pinned: false
license: mit
---

# Anti-Cheating AI ML System

HackerRank level anti-cheating system with comprehensive analysis pipelines.

## Features

### 🎥 Video Analysis Pipeline
- **Face Verification**: Match candidate with reference image using DeepFace
- **Gaze Tracking**: Detect suspicious eye movements and looking away
- **Head Pose Analysis**: Identify extreme head movements
- **Multiple Person Detection**: Flag presence of multiple people

### 🎵 Audio Analysis Pipeline  
- **Voice Activity Detection**: Analyze speech patterns and pauses
- **Pitch Analysis**: Detect voice modulation and multiple speakers
- **Background Noise**: Identify suspicious environmental sounds
- **Fraud Scoring**: Calculate audio-based fraud indicators

### 💻 Code Analysis Pipeline
- **AI Detection**: Identify AI-generated code patterns
- **Structure Analysis**: Analyze code complexity and patterns
- **Variable Naming**: Detect generic/suspicious naming patterns
- **Comment Analysis**: Flag overly verbose or AI-like documentation

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run Server
```bash
python app.py
```

### Test System
```bash
python test_client_updated.py
```

## API Usage

### 1. Interview Monitoring (Real-time)
```
POST /interview-monitor
```

**Parameters:**
- `video` (file): Live video stream for analysis
- `reference_image` (file): Candidate reference photo

**Response:**
```json
{
  "status": "success",
  "fraud_score": 5,
  "risk_level": "medium",
  "face_verification": {
    "verified": true,
    "confidence": 0.85
  },
  "mobile_detection": {
    "multiple_faces_detected": false,
    "suspicious_behavior_score": 2
  },
  "recommendations": ["Monitor closely"]
}
```

### 2. Post-Interview Analysis
```
POST /post-interview-analysis
```

**Parameters:**
- `audio` (file): Interview audio recording
- `code_text` (string): Submitted code solution

**Response:**
```json
{
  "status": "success",
  "overall_fraud_score": 8,
  "risk_level": "high",
  "audio_analysis": {...},
  "code_analysis": {...},
  "recommendations": ["Manual review required"]
}
```

## Risk Levels
- **Low** (0-4): No significant fraud indicators
- **Medium** (5-9): Additional verification suggested
- **High** (10-14): Detailed investigation recommended  
- **Critical** (15+): Immediate manual review required

## Supported Formats
- **Video**: MP4, AVI, MOV, MKV
- **Audio**: WAV, MP3, FLAC, M4A
- **Images**: JPG, JPEG, PNG, BMP