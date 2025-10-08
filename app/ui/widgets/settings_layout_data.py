from app.ui.widgets.actions import control_actions
import cv2
from app.helpers.typing_helper import LayoutDictTypes
SETTINGS_LAYOUT_DATA: LayoutDictTypes = {
    'Appearance': {
        'ThemeSelection': {
            'level': 1,
            'label': 'Theme',
            'options': ['Dark', 'Dark-Blue', 'Light'],
            'default': 'Dark',
            'help': 'Select the theme to be used',
            'exec_function': control_actions.change_theme,
            'exec_function_args': [],
        },
    },
    'General': {
        'ProvidersPrioritySelection': {
            'level': 1,
            'label': 'Providers Priority',
            'options': ['Auto', 'CUDA', 'TensorRT', 'TensorRT-Engine', 'CPU'],
            'default': 'Auto',
            'help': 'Select the providers priority to be used with the system.',
            'exec_function': control_actions.change_execution_provider,
            'exec_function_args': [],
        },
        'nThreadsSlider': {
            'level': 1,
            'label': 'Number of Threads',
            'min_value': '1',
            'max_value': '30',
            'default': '2',
            'step': 1,
            'help': 'Set number of execution threads while playing and recording. Depends strongly on GPU VRAM.',
            'exec_function': control_actions.change_threads_number,
            'exec_function_args': [],
        },
    },
    'Video Settings': {
        'VideoPlaybackCustomFpsToggle': {
            'level': 1,
            'label': 'Set Custom Video Playback FPS',
            'default': False,
            'help': 'Manually set the FPS to be used when playing the video',
            'exec_function': control_actions.set_video_playback_fps,
            'exec_function_args': [],
        },
        'VideoPlaybackCustomFpsSlider': {
            'level': 2,
            'label': 'Video Playback FPS',
            'min_value': '1',
            'max_value': '120',
            'default': '30',
            'parentToggle': 'VideoPlaybackCustomFpsToggle',
            'requiredToggleValue': True,
            'step': 1,
            'help': 'Set the maximum FPS of the video when playing'
        },
    },
    'Auto Swap':{
        'AutoSwapToggle': {
            'level': 1,
            'label': 'Auto Swap',
            'default': False,
            'help': 'Automatically Swap all faces using selected Source Faces/Embeddings when loading an video/image file'
        },
    },
    'Detectors': {
        'DetectorModelSelection': {
            'level': 1,
            'label': 'Face Detect Model',
            'options': ['RetinaFace', 'Yolov8', 'SCRFD', 'Yunet', 'MediaPipe'],
            'default': 'RetinaFace',
            'help': 'Select the face detection model to use for detecting faces in the input image or video. MediaPipe is optional and will be used only if the mediapipe package is available.'
        },
        'DetectorScoreSlider': {
            'level': 1,
            'label': 'Detect Score',
            'min_value': '1',
            'max_value': '100',
            'default': '50',
            'step': 1,
            'help': 'Set the confidence score threshold for face detection. Higher values ensure more confident detections but may miss some faces.'
        },
        'DetectionNMSThresholdSlider': {
            'level': 1,
            'label': 'NMS Threshold (%)',
            'min_value': '0',
            'max_value': '100',
            'default': '40',
            'step': 5,
            'help': 'Non-Maximum Suppression threshold controlling how much overlap is allowed between boxes before suppressing the lower score. Lower values remove more overlaps.'
        },
        'DetectionAutoRetryEnableToggle': {
            'level': 1,
            'label': 'Auto-Retry When No Face Found',
            'default': True,
            'help': 'If no face is detected, automatically retry with more rotations, lower score, and larger input size. Can slow down processing on hard frames.'
        },
        'DetectionLockToLastSwappedEnableToggle': {
            'level': 1,
            'label': 'Lock Detection To Last Swapped',
            'default': True,
            'help': 'Only consider detections near the last swapped face position for each target, to avoid swapping unintended faces.'
        },
        'DetectionForceNearPxSlider': {
            'level': 2,
            'label': 'Force Swap Radius (px)',
            'min_value': '0',
            'max_value': '50',
            'default': '6',
            'step': 1,
            'parentToggle': 'DetectionLockToLastSwappedEnableToggle',
            'requiredToggleValue': True,
            'help': 'If a detection is within this many pixels of the last swapped center for the same target, force-apply the swap even if similarity is below threshold.'
        },
        'DetectionLockRadiusPercentSlider': {
            'level': 2,
            'label': 'Lock Radius (%) of Diagonal',
            'min_value': '0',
            'max_value': '5',
            'default': '1',
            'step': 1,
            'parentToggle': 'DetectionLockToLastSwappedEnableToggle',
            'requiredToggleValue': True,
            'help': 'Distance from last swapped center within which detections are considered. Very small values keep swaps tightly constrained to the previous face location.'
        },
        'DetectionLockEaseRadiusPercentSlider': {
            'level': 2,
            'label': 'Near Radius For Easing (%)',
            'min_value': '0',
            'max_value': '5',
            'default': '0',
            'step': 1,
            'parentToggle': 'DetectionLockToLastSwappedEnableToggle',
            'requiredToggleValue': True,
            'help': 'Within this smaller radius (integer % of diagonal), temporarily lower similarity threshold a few points to recover borderline matches without affecting other faces.'
        },
        'DetectionLockEasePointsSlider': {
            'level': 2,
            'label': 'Easing (points)',
            'min_value': '0',
            'max_value': '20',
            'default': '4',
            'step': 1,
            'parentToggle': 'DetectionLockToLastSwappedEnableToggle',
            'requiredToggleValue': True,
            'help': 'How many points to lower the similarity threshold within the near radius.'
        },
        'MaxFacesToDetectSlider': {
            'level': 1,
            'label': 'Max No of Faces to Detect',
            'min_value': '1',
            'max_value': '50',
            'default': '20',
            'step': 1,     
            'help': 'Set the maximum number of faces to detect in a frame'
   
        },
        'AutoRotationToggle': {
            'level': 1,
            'label': 'Auto Rotation',
            'default': False,
            'help': 'Automatically rotate the input to detect faces in various orientations.'
        },
        'ManualRotationEnableToggle': {
            'level': 1,
            'label': 'Manual Rotation',
            'default': False,
            'help': 'Rotate the face detector to better detect faces at different angles.'
        },
        'ManualRotationAngleSlider': {
            'level': 2,
            'label': 'Rotation Angle',
            'min_value': '0',
            'max_value': '270',
            'default': '0',
            'step': 90,
            'parentToggle': 'ManualRotationEnableToggle',
            'requiredToggleValue': True,
            'help': 'Set this to the angle of the input face angle to help with laying down/upside down/etc. Angles are read clockwise.'
        },
        'LandmarkDetectToggle': {
            'level': 1,
            'label': 'Enable Landmark Detection',
            'default': False,
            'help': 'Enable or disable facial landmark detection, which is used to refine face alignment.'
        },
        'LandmarkDetectModelSelection': {
            'level': 2,
            'label': 'Landmark Detect Model',
            'options': ['5', '68', '3d68', '98', '106', '203', '478'],
            'default': '203',
            'parentToggle': 'LandmarkDetectToggle',
            'requiredToggleValue': True,
            'help': 'Select the landmark detection model, where different models detect varying numbers of facial landmarks.'
        },
        'LandmarkDetectScoreSlider': {
            'level': 2,
            'label': 'Landmark Detect Score',
            'min_value': '1',
            'max_value': '100',
            'default': '50',
            'step': 1,
            'parentToggle': 'LandmarkDetectToggle',
            'requiredToggleValue': True,
            'help': 'Set the confidence score threshold for facial landmark detection.'
        },
        'DetectFromPointsToggle': {
            'level': 2,
            'label': 'Detect From Points',
            'default': False,
            'parentToggle': 'LandmarkDetectToggle',
            'requiredToggleValue': True,
            'help': 'Enable detection of faces from specified landmark points.'
        },
        'ShowLandmarksEnableToggle': {
            'level': 1,
            'label': 'Show Landmarks',
            'default': False,
            'help': 'Show Landmarks in realtime.'
        },
        'ShowAllDetectedFacesBBoxToggle': {
            'level': 1,
            'label': 'Show Bounding Boxes',
            'default': False,
            'help': 'Draw bounding boxes to all detected faces in the frame'
        },
        'DetectionFallbackMaxFramesSlider': {
            'level': 1,
            'label': 'Detection Fallback Frames',
            'min_value': '0',
            'max_value': '10',
            'default': '3',
            'step': 1,
            'help': 'Maximum number of consecutive frames to keep swapping using LK tracking when detection fails.'
        },
        'DetectionRedetectROIToggle': {
            'level': 1,
            'label': 'ROI Re-detect after Tracking',
            'default': True,
            'help': 'After using tracking for a frame, try a focused re-detection on a padded ROI to break long tracking chains.'
        },
        'DetectionRedetectROIPaddingSlider': {
            'level': 2,
            'label': 'ROI Padding (%)',
            'min_value': '0',
            'max_value': '100',
            'default': '35',
            'step': 5,
            'parentToggle': 'DetectionRedetectROIToggle',
            'requiredToggleValue': True,
            'help': 'Percentage padding around the tracked union bbox for the ROI re-detection.'
        },
        'DetectionDeduplicateEnableToggle': {
            'level': 1,
            'label': 'Suppress Duplicate Detections',
            'default': True,
            'help': 'Merge overlapping detections that belong to the same face to avoid double swapping.'
        },
        'DetectionDeduplicateIoUSlider': {
            'level': 2,
            'label': 'Dedup IoU Threshold (%)',
            'min_value': '20',
            'max_value': '90',
            'default': '55',
            'step': 5,
            'parentToggle': 'DetectionDeduplicateEnableToggle',
            'requiredToggleValue': True,
            'help': 'If overlap between two boxes exceeds this IoU threshold, keep only one (higher confidence).'
        },
        'DetectionDeduplicateCenterPctSlider': {
            'level': 2,
            'label': 'Dedup Center Distance (%)',
            'min_value': '5',
            'max_value': '50',
            'default': '18',
            'step': 1,
            'parentToggle': 'DetectionDeduplicateEnableToggle',
            'requiredToggleValue': True,
            'help': 'If two detections have centers closer than this percent of the bbox diagonal, treat as duplicates.'
        },
        'DetectionTemporalSmoothingEnableToggle': {
            'level': 1,
            'label': 'Detection Temporal Smoothing',
            'default': True,
            'help': 'Reduce small jitter across frames by smoothing detected boxes and landmarks over time.'
        },
        'DetectionTemporalSmoothingStrengthSlider': {
            'level': 2,
            'label': 'Smoothing Strength (%)',
            'min_value': '0',
            'max_value': '90',
            'default': '40',
            'step': 5,
            'parentToggle': 'DetectionTemporalSmoothingEnableToggle',
            'requiredToggleValue': True,
            'help': 'Higher values increase temporal smoothing (more influence from previous frame). 0 disables smoothing effect.'
        },
        'MinFaceSizeFilterEnableToggle': {
            'level': 1,
            'label': 'Ignore Very Small Faces',
            'default': False,
            'help': 'Filter out tiny faces to reduce false positives and jitter from distant/background faces.'
        },
        'MinFaceSizePercentSlider': {
            'level': 2,
            'label': 'Min Face Size (% of diagonal)',
            'min_value': '0',
            'max_value': '20',
            'default': '3',
            'step': 1,
            'parentToggle': 'MinFaceSizeFilterEnableToggle',
            'requiredToggleValue': True,
            'help': 'Discard faces whose bounding box diagonal is smaller than this percent of the image diagonal.'
        }
    },
    'Rotation Stabilization': {
        'RotationStabilizationEnableToggle': {
            'level': 1,
            'label': 'Rotation Stabilization',
            'default': True,
            'help': 'Stabilize small roll (in-plane) rotations to reduce jitter and resampling blur.'
        },
        'RotationStabilizationPresetSelection': {
            'level': 2,
            'label': 'Smoothing Preset',
            'options': ['Low', 'Medium', 'High'],
            'default': 'Medium',
            'parentToggle': 'RotationStabilizationEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust temporal smoothing strength for roll angle: Low (subtle), Medium (balanced), High (strong).'
        },
        'RotationRollThresholdSlider': {
            'level': 2,
            'label': 'Roll Threshold (deg)',
            'min_value': '1',
            'max_value': '10',
            'default': '3',
            'step': 1,
            'parentToggle': 'RotationStabilizationEnableToggle',
            'requiredToggleValue': True,
            'help': 'Degrees of roll under which micro-rotations are suppressed (with hysteresis).'
        },
    },
    'DFM Settings':{
        'MaxDFMModelsSlider':{
            'level': 1,
            'label': 'Maximum DFM Models to use',
            'min_value': '1',
            'max_value': '5',
            'default': '1',
            'step': 1,
            'help': "Set the maximum number of DFM Models to keep in memory at a time. Set this based on your GPU's VRAM",
        }
    },
    'Frame Enhancer':{
        'FrameEnhancerEnableToggle':{
            'level': 1,
            'label': 'Enable Frame Enhancer',
            'default': False,
            'help': 'Enable frame enhancement for video inputs to improve visual quality.'
        },
        'FrameEnhancerTypeSelection':{
            'level': 2,
            'label': 'Frame Enhancer Type',
            'options': ['RealEsrgan-x2-Plus', 'RealEsrgan-x4-Plus', 'RealEsr-General-x4v3', 'BSRGan-x2', 'BSRGan-x4', 'UltraSharp-x4', 'UltraMix-x4', 'DDColor-Artistic', 'DDColor', 'DeOldify-Artistic', 'DeOldify-Stable', 'DeOldify-Video'],
            'default': 'RealEsrgan-x2-Plus',
            'parentToggle': 'FrameEnhancerEnableToggle',
            'requiredToggleValue': True,
            'help': 'Select the type of frame enhancement to apply, based on the content and resolution requirements.'
        },
        'FrameEnhancerBlendSlider': {
            'level': 2,
            'label': 'Blend',
            'min_value': '0',
            'max_value': '100',
            'default': '100',
            'step': 1,
            'parentToggle': 'FrameEnhancerEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blends the enhanced results back into the original frame.'
        },
    },
    'Webcam Settings': {
        'WebcamMaxNoSelection': {
            'level': 2,
            'label': 'Webcam Max No',
            'options': ['1', '2', '3', '4', '5', '6'],
            'default': '1',
            'help': 'Select the maximum number of webcam streams to allow for face swapping.'
        },
        'WebcamBackendSelection': {
            'level': 2,
            'label': 'Webcam Backend',
            'options': ['Default', 'DirectShow', 'MSMF', 'V4L', 'V4L2', 'GSTREAMER'],
            'default': 'Default',
            'help': 'Choose the backend for accessing webcam input.'
        },
        'WebcamMaxResSelection': {
            'level': 2,
            'label': 'Webcam Resolution',
            'options': ['480x360', '640x480', '1280x720', '1920x1080', '2560x1440', '3840x2160'],
            'default': '1280x720',
            'help': 'Select the maximum resolution for webcam input.'
        },
        'WebCamMaxFPSSelection': {
            'level': 2,
            'label': 'Webcam FPS',
            'options': ['23', '30', '60'],
            'default': '30',
            'help': 'Set the maximum frames per second (FPS) for webcam input.'
        },
    },
    'Virtual Camera': {
        'SendVirtCamFramesEnableToggle': {
            'level': 1,
            'label': 'Send Frames to Virtual Camera',
            'default': False,
            'help': 'Send the swapped video/webcam output to virtual camera for using in external applications',
            'exec_function': control_actions.toggle_virtualcam,
            'exec_function_args': [],
        },
        'VirtCamBackendSelection': {
            'level': 1,
            'label': 'Virtual Camera Backend',
            'options': ['obs', 'unitycapture'],
            'default': 'obs',
            'help': 'Choose the backend based on the Virtual Camera you have set up',
            'parentToggle': 'SendVirtCamFramesEnableToggle',
            'requiredToggleValue': True,
            'exec_function': control_actions.enable_virtualcam,
            'exec_funtion_args': [],
        },
    },
    'Face Recognition': {
        'RecognitionModelSelection': {
            'level': 1,
            'label': 'Recognition Model',
            'options': ['Inswapper128ArcFace', 'SimSwapArcFace', 'GhostArcFace', 'CSCSArcFace'],
            'default': 'Inswapper128ArcFace',
            'help': 'Choose the ArcFace model to be used for comparing the similarity of faces.'
        },
        'SimilarityTypeSelection': {
            'level': 1,
            'label': 'Swapping Similarity Type',
            'options': ['Opal', 'Pearl', 'Optimal'],
            'default': 'Opal',
            'help': 'Choose the type of similarity calculation for face detection and matching during the face swapping process.'
        },
    },
    'Embedding Merge Method':{
        'EmbMergeMethodSelection':{
            'level': 1,
            'label': 'Embedding Merge Method',
            'options': ['Mean','Median'],
            'default': 'Mean',
            'help': 'Select the method to merge facial embeddings. "Mean" averages the embeddings, while "Median" selects the middle value, providing more robustness to outliers.'
        }
    },
    'Media Selection':{
        'TargetMediaFolderRecursiveToggle':{
            'level': 1,
            'label': 'Target Media Include Subfolders',
            'default': False,
            'help': 'Include all files from Subfolders when choosing Target Media Folder'
        },
        'InputFacesFolderRecursiveToggle':{
            'level': 1,
            'label': 'Input Faces Include Subfolders',
            'default': False,
            'help': 'Include all files from Subfolders when choosing Input Faces Folder'
        }
    }
}

CAMERA_BACKENDS = {
    'Default': cv2.CAP_ANY,
    'DirectShow': cv2.CAP_DSHOW,
    'MSMF': cv2.CAP_MSMF,
    'V4L': cv2.CAP_V4L,
    'V4L2': cv2.CAP_V4L2,
    'GSTREAMER': cv2.CAP_GSTREAMER,
}