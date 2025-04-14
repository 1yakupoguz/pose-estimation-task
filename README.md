# Human Pose Detection and Tracking System

## 📋 Table of Contents (v1)
- [Overview](#overview)
- [Technical Components](#technical-components)
- [Technical Details](#technical-details)
- [Implementation Details](#implementation-details)
- [Results and Example Files](#results-and-example-files)

## Overview

This system uses computer vision and deep learning to detect and track human poses in videos. It identifies individuals, tracks their movements, and classifies their postures (standing, squatting, lying, running) over time, generating both a processed video output and analytical data in CSV format.

## Technical Components

### 1. Core Technologies

- **YOLO (You Only Look Once)**: YOLOv11s-pose model for human detection and pose estimation
- **OpenCV**: For video processing and visualization
- **NumPy**: For numerical operations and array manipulations
- **CSV**: For logging pose data

### 2. Object Detection and Pose Estimation

- Uses **YOLOv11s-pose** pre-trained model to:
  - Detect persons in each frame
  - Extract 17 keypoints representing body joints for each person
  - Generate bounding boxes for identified people

### 3. Multi-Object Tracking System

A custom tracking algorithm is implemented to:
- Maintain consistent IDs for each person across frames
- Track their center points using Euclidean distance (<50 pixel threshold)
- Store and update keypoints, bounding boxes, and pose statuses for each ID
- Handle both existing and new person detections

### 4. Pose Analysis

The system analyzes human poses through several techniques:

- **Joint Angle Calculation**: Computes angles between body segments using vector operations
- **Bounding Box Aspect Ratio Analysis**: Width-to-height ratio helps distinguish standing vs. lying postures
- **Keypoint Proximity Analysis**: Detects when wrists are near ankles (for squatting detection)
- **Velocity Calculation**: Tracks movement speed to detect running

### 5. Pose Classification Rules

- **Standing**: Default pose when person is upright (bbox ratio < 0.5)
- **Lying**: When bbox width > 1.5× height
- **Squatting**: When:
  - Average knee angle < threshold (default 130°), or
  - Wrist(s) are near ankle(s)
- **Running**: When average velocity exceeds threshold over multiple frames

### 6. Running Detection System

- Maintains a rolling window of velocity data (default 5 frames)
- Implements a counter-based confirmation system to avoid false positives
- Requires consistent high velocity for multiple frames to confirm running state

### 7. Data Management

- **Tracking Objects**: Dictionary of person IDs mapped to their center points
- **Tracking Keypoints**: Dictionary storing each person's joint coordinates
- **Tracking Boxes**: Dictionary of bounding boxes for each person
- **Pose Status**: Dictionary tracking each person's current pose and duration
- **Velocity Tracking**: Dictionary recording movement speeds for running detection

### 8. Visualization

- Person bounding boxes with unique color coding based on ID
- Person ID labels and center points
- Skeleton rendering connecting visible keypoints
- Keypoint visualization with index numbers
- Pose status text with duration counters
- Velocity metrics display

### 9. Output Generation

- **Processed Video**: Annotated video with all visual elements
- **CSV Log**: Detailed record of pose changes including:
  - Person ID
  - Pose type
  - Duration in seconds
  - Timestamp in video

## Technical Details

### Distance Thresholds

- **Tracking distance**: 50 pixels (for ID continuity)
- **Wrist-ankle proximity**: 30% of person height
- **Running speed threshold**: 10 (pixel displacement between frames)

### Performance Optimizations

- Pose analysis performed every 5 frames instead of every frame
- Keypoint filtering based on confidence score (>0.5) and bounding box containment
- Only visible keypoints are rendered in the skeleton visualization

### Key Functions

1. **`calculate_angle(a, b, c)`**: Computes the angle between three points using vector math
2. **`analyze_pose(keypoints, bbox, object_id, ...)`**: Determines the person's posture based on multiple criteria
3. **`calculate_velocity(prev_point, current_point)`**: Measures displacement between consecutive frames

### Data Flow

1. Frame acquisition from video source
2. YOLO model inference for person detection and keypoint extraction
3. Center point calculation and tracking ID assignment
4. Pose analysis and classification (every 5th frame)
5. Visualization generation with annotations
6. Frame display and recording to output video
7. Pose change logging for CSV export

## Implementation Notes

- The system performs pose analysis at 1/5 frame rate to reduce computational load
- Missing keypoints are handled gracefully by the pose analyzer
- The tracking system can accommodate people entering and leaving the frame
- Person IDs are maintained as long as the person remains visible
- For each pose change, duration is recorded before updating to new pose
- Final CSV includes completed activities plus any ongoing activities at video end

## Results and Example Files
- Sample outputs of the project, including processed videos and log files, can be found at the following Google Drive link:

[**Google Drive Link**](https://drive.google.com/drive/folders/1u8Xxafn2_XAXw-vduyOwAzgEflIgKET4?usp=sharing)

- In this folder, you will find:
  - Processed video files
  - Posture log records in CSV format
  - Example scenarios demonstrating system performance
  - You can review these files to observe the system's real-time operation and posture analysis results.