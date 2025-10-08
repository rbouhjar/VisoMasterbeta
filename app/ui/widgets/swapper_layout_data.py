from app.helpers import miscellaneous as misc_helpers
from app.ui.widgets.actions import layout_actions
from app.ui.widgets.actions import common_actions as common_widget_actions
from app.helpers.typing_helper import LayoutDictTypes

# Widgets in Face Swap tab are created from this Layout
SWAPPER_LAYOUT_DATA: LayoutDictTypes = {
    'Swapper': {
        'SwapModelSelection': {
            'level': 1,
            'label': 'Swapper Model',
            'options': ['Inswapper128', 'InStyleSwapper256 Version A', 'InStyleSwapper256 Version B', 'InStyleSwapper256 Version C', 'DeepFaceLive (DFM)', 'SimSwap512', 'GhostFace-v1', 'GhostFace-v2', 'GhostFace-v3', 'CSCS'],            'default': 'Inswapper128',
            'help': 'Choose which swapper model to use for face swapping.'
        },
        'SwapperResSelection': {
            'level': 2,
            'label': 'Swapper Resolution',
            'options': ['128', '256', '384', '512'],
            'default': '128',
            'parentSelection': 'SwapModelSelection',
            'requiredSelectionValue': 'Inswapper128',
            'help': 'Select the resolution for the swapped face in pixels. Higher values offer better quality but are slower to process.'
        },
        'DFMModelSelection': {
            'level': 2,
            'label': 'DFM Model',
            'options': misc_helpers.get_dfm_models_selection_values,
            'default': misc_helpers.get_dfm_models_default_value,
            'parentSelection': 'SwapModelSelection',
            'requiredSelectionValue': 'DeepFaceLive (DFM)',
            'help': 'Select which pretrained DeepFaceLive (DFM) Model to use for swapping.'
        },
        'DFMAmpMorphSlider': {
            'level': 2,
            'label': 'AMP Morph Factor',
            'min_value': '1',
            'max_value': '100',
            'default': '50',
            'step': 1,
            'parentSelection': 'SwapModelSelection',
            'requiredSelectionValue': 'DeepFaceLive (DFM)',
            'help': 'AMP Morph Factor for DFM AMP Models',
        },
        'DFMRCTColorToggle': {
            'level': 2,
            'label': 'RCT Color Transfer',
            'default': False,
            'parentSelection': 'SwapModelSelection',
            'requiredSelectionValue': 'DeepFaceLive (DFM)',
            'help': 'RCT Color Transfer for DFM Models',
        }
    },
    'AI Assist': {
        'AISmartTuningEnableToggle': {
            'level': 1,
            'label': 'AI Smart Tuning',
            'default': False,
            'help': 'Automatically adjust key settings per-frame based on motion, lighting, face size and pose. Changes are transient and do not overwrite your saved defaults.'
        },
        'AISmartTuningModeSelection': {
            'level': 2,
            'label': 'Mode',
            'options': ['Conservative', 'Balanced', 'Aggressive'],
            'default': 'Balanced',
            'parentToggle': 'AISmartTuningEnableToggle',
            'requiredToggleValue': True,
            'help': 'Conservative = subtle adjustments, Balanced = recommended, Aggressive = stronger assistance in difficult conditions.'
        },
        'AISmartTuningReactivitySlider': {
            'level': 2,
            'label': 'Reactivity (frames)',
            'min_value': '1',
            'max_value': '60',
            'default': '12',
            'step': 1,
            'parentToggle': 'AISmartTuningEnableToggle',
            'requiredToggleValue': True,
            'help': 'How quickly the tuning reacts to changes. Lower = more reactive, higher = smoother.'
        },
        'AIAgentEnableToggle': {
            'level': 1,
            'label': 'AI Preset Agent (auto profiles)',
            'default': False,
            'help': 'Automatically selects one of the ready-made profiles (Portrait Stable, Vlog Mobile, Low-Light) based on motion, lighting and face size. Applies presets with a cooldown to avoid flicker.'
        },
        'AIAgentMinDwellSecondsSlider': {
            'level': 2,
            'label': 'Agent: Min Dwell (s)',
            'min_value': '1',
            'max_value': '20',
            'default': '6',
            'step': 1,
            'parentToggle': 'AIAgentEnableToggle',
            'requiredToggleValue': True,
            'help': 'Minimum time in seconds to keep a chosen profile before the agent is allowed to switch again.'
        },
        'AIAgentDebugOverlayEnableToggle': {
            'level': 2,
            'label': 'Agent: Show Debug Overlay',
            'default': False,
            'parentToggle': 'AIAgentEnableToggle',
            'requiredToggleValue': True,
            'help': 'Show agent-selected profile and scene metrics on screen.'
        },
    },
    'Face Landmarks Correction': {
        'FaceAdjEnableToggle': {
            'level': 1,
            'label': 'Face Adjustments',
            'default': False,
            'help': 'This is an experimental feature to perform direct adjustments to the face landmarks found by the detector. There is also an option to adjust the scale of the swapped face.'
        },
        'KpsXSlider': {
            'level': 2,
            'label': 'Keypoints X-Axis',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceAdjEnableToggle',
            'requiredToggleValue': True,
            'help': 'Shifts the detection points left and right.'
        },
        'KpsYSlider': {
            'level': 2,
            'label': 'Keypoints Y-Axis',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceAdjEnableToggle',
            'requiredToggleValue': True,
            'help': 'Shifts the detection points up and down.'
        },
        'KpsScaleSlider': {
            'level': 2,
            'label': 'Keypoints Scale',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceAdjEnableToggle',
            'requiredToggleValue': True,
            'help': 'Grows and shrinks the detection point distances.'
        },
        'FaceScaleAmountSlider': {
            'level': 2,
            'label': 'Face Scale Amount',
            'min_value': '-20',
            'max_value': '20',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceAdjEnableToggle',
            'requiredToggleValue': True,
            'help': 'Grows and shrinks the entire face.'
        },
        'LandmarksPositionAdjEnableToggle': {
            'level': 1,
            'label': '5 - Keypoints Adjustments',
            'default': False,
            'help': 'This is an experimental feature to perform direct adjustments to the position of face landmarks found by the detector.'
        },
        'EyeLeftXAmountSlider': {
            'level': 2,
            'label': 'Left Eye:   X',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'LandmarksPositionAdjEnableToggle',
            'requiredToggleValue': True,
            'help': 'Shifts the eye left detection point left and right.'
        },
        'EyeLeftYAmountSlider': {
            'level': 2,
            'label': 'Left Eye:   Y',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'LandmarksPositionAdjEnableToggle',
            'requiredToggleValue': True,
            'help': 'Shifts the eye left detection point up and down.'
        },
        'EyeRightXAmountSlider': {
            'level': 2,
            'label': 'Right Eye:   X',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'LandmarksPositionAdjEnableToggle',
            'requiredToggleValue': True,
            'help': 'Shifts the eye right detection point left and right.'
        },
        'EyeRightYAmountSlider': {
            'level': 2,
            'label': 'Right Eye:   Y',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'LandmarksPositionAdjEnableToggle',
            'requiredToggleValue': True,
            'help': 'Shifts the eye right detection point up and down.'
        },
        'NoseXAmountSlider': {
            'level': 2,
            'label': 'Nose:   X',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'LandmarksPositionAdjEnableToggle',
            'requiredToggleValue': True,
            'help': 'Shifts the nose detection point left and right.'
        },
        'NoseYAmountSlider': {
            'level': 2,
            'label': 'Nose:   Y',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'LandmarksPositionAdjEnableToggle',
            'requiredToggleValue': True,
            'help': 'Shifts the nose detection point up and down.'
        },
        'MouthLeftXAmountSlider': {
            'level': 2,
            'label': 'Left Mouth:   X',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'LandmarksPositionAdjEnableToggle',
            'requiredToggleValue': True,
            'help': 'Shifts the mouth left detection point left and right.'
        },
        'MouthLeftYAmountSlider': {
            'level': 2,
            'label': 'Left Mouth:   Y',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'LandmarksPositionAdjEnableToggle',
            'requiredToggleValue': True,
            'help': 'Shifts the mouth left detection point up and down.'
        },
        'MouthRightXAmountSlider': {
            'level': 2,
            'label': 'Right Mouth:   X',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'LandmarksPositionAdjEnableToggle',
            'requiredToggleValue': True,
            'help': 'Shifts the mouth Right detection point left and right.'
        },
        'MouthRightYAmountSlider': {
            'level': 2,
            'label': 'Right Mouth:   Y',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'LandmarksPositionAdjEnableToggle',
            'requiredToggleValue': True,
            'help': 'Shifts the mouth Right detection point up and down.'
        },
    },
    'Face Similarity': {
        'SimilarityThresholdSlider': {
            'level': 1,
            'label': 'Similarity Threshold',
            'min_value': '1',
            'max_value': '100',
            'default': '60',
            'step': 1,
            'help': 'Set the similarity threshold to control how similar the detected face should be to the reference (target) face.'
        },
        'SimilarityLocalityEnableToggle': {
            'level': 1,
            'label': 'Localize Impact Near Face',
            'default': False,
            'help': 'When enabled, the swap impact is constrained to a region near the face center to keep changes tightly focused.'
        },
        'SimilarityLocalityRadiusPercentSlider': {
            'level': 2,
            'label': 'Local Radius (%)',
            'min_value': '5',
            'max_value': '100',
            'default': '35',
            'step': 5,
            'parentToggle': 'SimilarityLocalityEnableToggle',
            'requiredToggleValue': True,
            'help': 'Radius of the localized impact, as a percentage of the face crop size. Smaller values restrict the effect to the core facial region.'
        },
        'SimilarityLocalityFeatherPercentSlider': {
            'level': 2,
            'label': 'Local Feather (%)',
            'min_value': '0',
            'max_value': '100',
            'default': '30',
            'step': 5,
            'parentToggle': 'SimilarityLocalityEnableToggle',
            'requiredToggleValue': True,
            'help': 'Feathering at the edge of the localized region for a smooth transition.'
        },
        'AutoThresholdAssistEnableToggle': {
            'level': 1,
            'label': 'Auto Threshold Assist',
            'default': False,
            'help': 'When enabled, slightly relaxes the similarity threshold on sustained no-match streaks and resets after a successful swap.'
        },
        'AutoThresholdAssistMaxDropSlider': {
            'level': 2,
            'label': 'Assist: Max Threshold Drop',
            'min_value': '0',
            'max_value': '30',
            'default': '10',
            'step': 1,
            'parentToggle': 'AutoThresholdAssistEnableToggle',
            'requiredToggleValue': True,
            'help': 'Maximum number of points the effective similarity threshold can be lowered (bounded).'
        },
        'AutoThresholdAssistFramesPerStepSlider': {
            'level': 2,
            'label': 'Assist: Frames per Step',
            'min_value': '1',
            'max_value': '60',
            'default': '15',
            'step': 1,
            'parentToggle': 'AutoThresholdAssistEnableToggle',
            'requiredToggleValue': True,
            'help': 'Number of consecutive no-match frames required to lower the threshold by one point.'
        },
        'StrengthEnableToggle': {
            'level': 1,
            'label': 'Strength',
            'default': False,
            'help': 'Apply additional swapping iterations to increase the strength of the result, which may increase likeness.'
        },
        'StrengthAmountSlider': {
            'level': 2,
            'label': 'Amount',
            'min_value': '0',
            'max_value': '500',
            'default': '100',
            'step': 25,
            'parentToggle': 'StrengthEnableToggle',
            'requiredToggleValue': True,
            'help': 'Increase up to 5x additional swaps (500%). 200% is generally a good result. Set to 0 to turn off swapping but allow the rest of the pipeline to apply to the original image.'
        },
        'FaceLikenessEnableToggle': {
            'level': 1,
            'label': 'Face Likeness',
            'default': False,
            'help': 'This is a feature to perform direct adjustments to likeness of faces.'
        },
        'FaceLikenessFactorDecimalSlider': {
            'level': 2,
            'label': 'Amount',
            'min_value': '-1.00',
            'max_value': '1.00',
            'default': '0.00',
            'decimals': 2,
            'step': 0.05,
            'parentToggle': 'FaceLikenessEnableToggle',
            'requiredToggleValue': True,
            'help': 'Determines the factor of likeness between the source and assigned faces.'
        },
        'DifferencingEnableToggle': {
            'level': 1,
            'label': 'Differencing',
            'default': False,
            'help': 'Allow some of the original face to show in the swapped result when the difference between the two images is small. Can help bring back some texture to the swapped face.'
        },
        'DifferencingAmountSlider': {
            'level': 2,
            'label': 'Amount',
            'min_value': '0',
            'max_value': '100',
            'default': '4',
            'step': 1,
            'parentToggle': 'DifferencingEnableToggle',
            'requiredToggleValue': True,
            'help': 'Higher values relaxes the similarity constraint.'
        },
        'DifferencingBlendAmountSlider': {
            'level': 2,
            'label': 'Blend Amount',
            'min_value': '0',
            'max_value': '100',
            'default': '5',
            'step': 1,
            'parentToggle': 'DifferencingEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blend differecing value.'
        },
        'RecognitionDebugOverlayEnableToggle': {
            'level': 1,
            'label': 'Show Recognition Debug Overlay',
            'default': False,
            'help': 'Displays top similarity and thresholds on-screen to help tune recognition.'
        },
    },
    'Face Mask':{
        'BorderBottomSlider':{
            'level': 1,
            'label': 'Bottom Border',
            'min_value': '0',
            'max_value': '100',
            'default': '10',
            'step': 1,
            'help': 'A rectangle with adjustable bottom, left, right, top, and sides that masks the swapped face result back into the original image.'
        },
        'BorderLeftSlider':{
            'level': 1,
            'label': 'Left Border',
            'min_value': '0',
            'max_value': '100',
            'default': '10',
            'step': 1,
            'help': 'A rectangle with adjustable bottom, left, right, top, and sides that masks the swapped face result back into the original image.'
        },
        'BorderRightSlider':{
            'level': 1,
            'label': 'Right Border',
            'min_value': '0',
            'max_value': '100',
            'default': '10',
            'step': 1,
            'help': 'A rectangle with adjustable bottom, left, right, top, and sides that masks the swapped face result back into the original image.'
        },
        'BorderTopSlider':{
            'level': 1,
            'label': 'Top Border',
            'min_value': '0',
            'max_value': '100',
            'default': '10',
            'step': 1,
            'help': 'A rectangle with adjustable bottom, left, right, top, and sides that masks the swapped face result back into the original image.'
        },
        'BorderBlurSlider':{
            'level': 1,
            'label': 'Border Blur',
            'min_value': '0',
            'max_value': '100',
            'default': '10',
            'step': 1,
            'help': 'Border mask blending distance.'
        },
        'OccluderEnableToggle': {
            'level': 1,
            'label': 'Occlusion Mask',
            'default': False,
            'help': 'Allow objects occluding the face to show up in the swapped image.'
        },
        'OccluderSizeSlider': {
            'level': 2,
            'label': 'Size',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'OccluderEnableToggle',
            'requiredToggleValue': True,
            'help': 'Grows or shrinks the occluded region'
        },
        'DFLXSegEnableToggle': {
            'level': 1,
            'label': 'DFL XSeg Mask',
            'default': False,
            'help': 'Allow objects occluding the face to show up in the swapped image.'
        },
        'DFLXSegSizeSlider': {
            'level': 2,
            'label': 'Size',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'DFLXSegEnableToggle',
            'requiredToggleValue': True,
            'help': 'Grows or shrinks the occluded region.'
        },
        'OccluderXSegBlurSlider': {
            'level': 1,
            'label': 'Occluder/DFL XSeg Blur',
            'min_value': '0',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'OccluderEnableToggle | DFLXSegEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blend value for Occluder and XSeg.'
        },
        'ClipEnableToggle': {
            'level': 1,
            'label': 'Text Masking',
            'default': False,
            'help': 'Use descriptions to identify objects that will be present in the final swapped image.'
        },
        'ClipText': {
            'level': 2,
            'label': 'Text Masking Entry',
            'min_value': '0',
            'max_value': '1000',
            'default': '',
            'width': 130,
            'parentToggle': 'ClipEnableToggle',
            'requiredToggleValue': True,
            'help': 'To use, type a word(s) in the box separated by commas and press <enter>.'
        },
        'ClipAmountSlider': {
            'level': 2,
            'label': 'Amount',
            'min_value': '0',
            'max_value': '100',
            'default': '50',
            'step': 1,
            'parentToggle': 'ClipEnableToggle',
            'requiredToggleValue': True,
            'help': 'Increase to strengthen the effect.'
        },
        'FaceParserEnableToggle': {
            'level': 1,
            'label': 'Face Parser Mask',
            'default': False,
            'help': 'Allow the unprocessed background from the orginal image to show in the final swap.'
        },
        'BackgroundParserSlider': {
            'level': 2,
            'label': 'Background',
            'min_value': '-50',
            'max_value': '50',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Negative/Positive values shrink and grow the mask.'
        },
        'FaceParserSlider': {
            'level': 2,
            'label': 'Face',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the size of the Mask. Mast the entire face.'
        },
        'LeftEyebrowParserSlider': {
            'level': 2,
            'label': 'Left Eyebrow',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the size of the Mask. Mast the left eyebrow.'
        },
        'RightEyebrowParserSlider': {
            'level': 2,
            'label': 'Right Eyebrow',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the size of the Mask. Mast the right eyebrow.'
        },
        'LeftEyeParserSlider': {
            'level': 2,
            'label': 'Left Eye',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the size of the Mask. Mast the left eye.'
        },
        'RightEyeParserSlider': {
            'level': 2,
            'label': 'Right Eye',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the size of the Mask. Mast the right eye.'
        },
        'EyeGlassesParserSlider': {
            'level': 2,
            'label': 'EyeGlasses',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the size of the Mask. Mast the eyeglasses.'
        },
        'NoseParserSlider': {
            'level': 2,
            'label': 'Nose',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the size of the Mask. Mast the nose.'
        },
        'MouthParserSlider': {
            'level': 2,
            'label': 'Mouth',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the size of the Mask. Mast the inside of the mouth, including the tongue.'
        },
        'UpperLipParserSlider': {
            'level': 2,
            'label': 'Upper Lip',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the size of the Mask. Mast the upper lip.'
        },
        'LowerLipParserSlider': {
            'level': 2,
            'label': 'Lower Lip',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the size of the Mask. Mast the lower lip.'
        },
        'NeckParserSlider': {
            'level': 2,
            'label': 'Neck',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the size of the Mask. Mast the neck.'
        },
        'HairParserSlider': {
            'level': 2,
            'label': 'Hair',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the size of the Mask. Mast the hair.'
        },
        'BackgroundBlurParserSlider': {
            'level': 2,
            'label': 'Background Blur',
            'min_value': '0',
            'max_value': '100',
            'default': '5',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blend the value for Background Parser'
        },
        'FaceBlurParserSlider': {
            'level': 2,
            'label': 'Face Blur',
            'min_value': '0',
            'max_value': '100',
            'default': '5',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blend the value for Face Parser'
        },
        'FaceParserHairMakeupEnableToggle': {
            'level': 2,
            'label': 'Hair Makeup',
            'default': False,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Enable hair makeup'
        },
        'FaceParserHairMakeupRedSlider': {
            'level': 3,
            'label': 'Red',
            'min_value': '0',
            'max_value': '255',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle & FaceParserHairMakeupEnableToggle',
            'requiredToggleValue': True,
            'help': 'Red color adjustments'
        },
        'FaceParserHairMakeupGreenSlider': {
            'level': 3,
            'label': 'Green',
            'min_value': '0',
            'max_value': '255',
            'default': '0',
            'step': 3,
            'parentToggle': 'FaceParserEnableToggle & FaceParserHairMakeupEnableToggle',
            'requiredToggleValue': True,
            'help': 'Green color adjustments'
        },
        'FaceParserHairMakeupBlueSlider': {
            'level': 3,
            'label': 'Blue',
            'min_value': '0',
            'max_value': '255',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle & FaceParserHairMakeupEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blue color adjustments'
        },
        'FaceParserHairMakeupBlendAmountDecimalSlider': {
            'level': 3,
            'label': 'Blend Amount',
            'min_value': '0.1',
            'max_value': '1.0',
            'default': '0.2',
            'step': 0.1,
            'decimals': 1,
            'parentToggle': 'FaceParserEnableToggle & FaceParserHairMakeupEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blend the value: 0.0 represents the original color, 1.0 represents the full target color.'
        },
        'FaceParserLipsMakeupEnableToggle': {
            'level': 2,
            'label': 'Lips Makeup',
            'default': False,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Enable lips makeup'
        },
        'FaceParserLipsMakeupRedSlider': {
            'level': 3,
            'label': 'Red',
            'min_value': '0',
            'max_value': '255',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle & FaceParserLipsMakeupEnableToggle',
            'requiredToggleValue': True,
            'help': 'Red color adjustments'
        },
        'FaceParserLipsMakeupGreenSlider': {
            'level': 3,
            'label': 'Green',
            'min_value': '0',
            'max_value': '255',
            'default': '0',
            'step': 3,
            'parentToggle': 'FaceParserEnableToggle & FaceParserLipsMakeupEnableToggle',
            'requiredToggleValue': True,
            'help': 'Green color adjustments'
        },
        'FaceParserLipsMakeupBlueSlider': {
            'level': 3,
            'label': 'Blue',
            'min_value': '0',
            'max_value': '255',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle & FaceParserLipsMakeupEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blue color adjustments'
        },
        'FaceParserLipsMakeupBlendAmountDecimalSlider': {
            'level': 3,
            'label': 'Blend Amount',
            'min_value': '0.1',
            'max_value': '1.0',
            'default': '0.2',
            'step': 0.1,
            'decimals': 1,
            'parentToggle': 'FaceParserEnableToggle & FaceParserLipsMakeupEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blend the value: 0.0 represents the original color, 1.0 represents the full target color.'
        },
        'RestoreEyesEnableToggle': {
            'level': 1,
            'label': 'Restore Eyes',
            'default': False,
            'help': 'Restore eyes from the original face.'
        },
        'RestoreEyesBlendAmountSlider': {
            'level': 2,
            'label': 'Eyes Blend Amount',
            'min_value': '1',
            'max_value': '100',
            'default': '50',
            'step': 1,
            'parentToggle': 'RestoreEyesEnableToggle',
            'requiredToggleValue': True,
            'help': 'Increase this to show more of the swapped eyes. Decrease it to show more of the original eyes.'
        },
        'RestoreEyesSizeFactorDecimalSlider': {
            'level': 2,
            'label': 'Eyes Size Factor',
            'min_value': '2.0',
            'max_value': '4.0',
            'default': '3.0',
            'decimals': 1,
            'step': 0.5,
            'parentToggle': 'RestoreEyesEnableToggle',
            'requiredToggleValue': True,
            'help': 'Reduce this when swapping faces zoomed out of the frame.'
        },
        'RestoreEyesFeatherBlendSlider': {
            'level': 2,
            'label': 'Eyes Feather Blend',
            'min_value': '1',
            'max_value': '100',
            'default': '10',
            'step': 1,
            'parentToggle': 'RestoreEyesEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the blending of eyes border. Increase this to show more of the original eyes. Decrease this to show more of the swapped eyes.'
        },
        'RestoreXEyesRadiusFactorDecimalSlider': {
            'level': 2,
            'label': 'X Eyes Radius Factor',
            'min_value': '0.3',
            'max_value': '3.0',
            'default': '1.0',
            'decimals': 1,
            'step': 0.1,
            'parentToggle': 'RestoreEyesEnableToggle',
            'requiredToggleValue': True,
            'help': 'These parameters determine the shape of the mask. If both are equal to 1.0, the mask will be circular. If either one is greater or less than 1.0, the mask will become oval, stretching or shrinking along the corresponding direction.'
        },
        'RestoreYEyesRadiusFactorDecimalSlider': {
            'level': 2,
            'label': 'Y Eyes Radius Factor',
            'min_value': '0.3',
            'max_value': '3.0',
            'default': '1.0',
            'decimals': 1,
            'step': 0.1,
            'parentToggle': 'RestoreEyesEnableToggle',
            'requiredToggleValue': True,
            'help': 'These parameters determine the shape of the mask. If both are equal to 1.0, the mask will be circular. If either one is greater or less than 1.0, the mask will become oval, stretching or shrinking along the corresponding direction.'
        },
        'RestoreXEyesOffsetSlider': {
            'level': 2,
            'label': 'X Eyes Offset',
            'min_value': '-300',
            'max_value': '300',
            'default': '0',
            'step': 1,
            'parentToggle': 'RestoreEyesEnableToggle',
            'requiredToggleValue': True,
            'help': 'Moves the Eyes Mask on the X Axis.'
        },
        'RestoreYEyesOffsetSlider': {
            'level': 2,
            'label': 'Y Eyes Offset',
            'min_value': '-300',
            'max_value': '300',
            'default': '0',
            'step': 1,
            'parentToggle': 'RestoreEyesEnableToggle',
            'requiredToggleValue': True,
            'help': 'Moves the Eyes Mask on the Y Axis.'
        },
        'RestoreEyesSpacingOffsetSlider': {
            'level': 2,
            'label': 'Eyes Spacing Offset',
            'min_value': '-200',
            'max_value': '200',
            'default': '0',
            'step': 1,
            'parentToggle': 'RestoreEyesEnableToggle',
            'requiredToggleValue': True,
            'help': 'Change the Eyes Spacing distance.'
        },
        'RestoreMouthEnableToggle': {
            'level': 1,
            'label': 'Restore Mouth',
            'default': False,
            'help': 'Restore mouth from the original face.'
        },
        'RestoreMouthBlendAmountSlider': {
            'level': 2,
            'label': 'Mouth Blend Amount',
            'min_value': '1',
            'max_value': '100',
            'default': '50',
            'step': 1,
            'parentToggle': 'RestoreMouthEnableToggle',
            'requiredToggleValue': True,
            'help': 'Increase this to show more of the swapped Mouth. Decrease it to show more of the original Mouth.'
        },       
        'RestoreMouthSizeFactorSlider': {
            'level': 2,
            'label': 'Mouth Size Factor',
            'min_value': '5',
            'max_value': '60',
            'default': '25',
            'step': 5,
            'parentToggle': 'RestoreMouthEnableToggle',
            'requiredToggleValue': True,
            'help': 'Increase this when swapping faces zoomed out of the frame.'
        },
        'RestoreMouthFeatherBlendSlider': {
            'level': 2,
            'label': 'Mouth Feather Blend',
            'min_value': '1',
            'max_value': '100',
            'default': '10',
            'step': 1,
            'parentToggle': 'RestoreMouthEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the border of Mouth blending. Increase this to show more of the original Mouth. Decrease this to show more of the swapped Mouth.'
        },
        'RestoreXMouthRadiusFactorDecimalSlider': {
            'level': 2,
            'label': 'X Mouth Radius Factor',
            'min_value': '0.3',
            'max_value': '3.0',
            'default': '1.0',
            'decimals': 1,
            'step': 0.1,
            'parentToggle': 'RestoreMouthEnableToggle',
            'requiredToggleValue': True,
            'help': 'These parameters determine the shape of the mask. If both are equal to 1.0, the mask will be circular. If either one is greater or less than 1.0, the mask will become oval, stretching or shrinking along the corresponding direction.'
        },
        'RestoreYMouthRadiusFactorDecimalSlider': {
            'level': 2,
            'label': 'Y Mouth Radius Factor',
            'min_value': '0.3',
            'max_value': '3.0',
            'default': '1.0',
            'decimals': 1,
            'step': 0.1,
            'parentToggle': 'RestoreMouthEnableToggle',
            'requiredToggleValue': True,
            'help': 'These parameters determine the shape of the mask. If both are equal to 1.0, the mask will be circular. If either one is greater or less than 1.0, the mask will become oval, stretching or shrinking along the corresponding direction.'
        },
        'RestoreXMouthOffsetSlider': {
            'level': 2,
            'label': 'X Mouth Offset',
            'min_value': '-300',
            'max_value': '300',
            'default': '0',
            'step': 1,
            'parentToggle': 'RestoreMouthEnableToggle',
            'requiredToggleValue': True,
            'help': 'Moves the Mouth Mask on the X Axis.'
        },
        'RestoreYMouthOffsetSlider': {
            'level': 2,
            'label': 'Y Mouth Offset',
            'min_value': '-300',
            'max_value': '300',
            'default': '0',
            'step': 1,
            'parentToggle': 'RestoreMouthEnableToggle',
            'requiredToggleValue': True,
            'help': 'Moves the Mouth Mask on the Y Axis.'
        },
        'RestoreEyesMouthBlurSlider': {
            'level': 1,
            'label': 'Eyes/Mouth Blur',
            'min_value': '0',
            'max_value': '50',
            'default': '0',
            'step': 1,
            'parentToggle': 'RestoreEyesEnableToggle | RestoreMouthEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the blur of mask border.'
        },
        'AutoPitchCompEnableToggle': {
            'level': 1,
            'label': 'Auto Pitch Compensation (Mask)',
            'default': False,
            'help': 'Adapte le blend du masque selon l\'inclinaison (tête vers le bas/haut).'
        },
        'AutoPitchCompStrengthSlider': {
            'level': 2,
            'label': 'Pitch Compensation Strength',
            'min_value': '0',
            'max_value': '100',
            'default': '40',
            'step': 5,
            'parentToggle': 'AutoPitchCompEnableToggle',
            'requiredToggleValue': True,
            'help': 'Force d\'atténuation directionnelle (haut/bas) près du bord du masque.'
        },
    },
    
    'Face Color Correction':{
        'AutoColorEnableToggle': {
            'level': 1,
            'label': 'AutoColor Transfer',
            'default': False,
            'help': 'Enable AutoColor Transfer: 1. Hans Test without mask, 2. Hans Test with mask, 3. DFL Method without mask, 4. DFL Original Method.'
        },
        'AutoColorTransferTypeSelection':{
            'level': 2,
            'label': 'Transfer Type',
            'options': ['Test', 'Test_Mask', 'DFL_Test', 'DFL_Orig'],
            'default': 'Test',
            'parentToggle': 'AutoColorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Select the AutoColor transfer method type. Hans Method could have some artefacts sometimes.'
        },
        'AutoColorBlendAmountSlider': {
            'level': 1,
            'label': 'Blend Amount',
            'min_value': '0',
            'max_value': '100',
            'default': '80',
            'step': 5,
            'parentToggle': 'AutoColorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the blend value.'
        },
        'ColorEnableToggle': {
            'level': 1,
            'label': 'Color Adjustments',
            'default': False,
            'help': 'Fine-tune the RGB color values of the swap.'
        },
        'ColorRedSlider': {
            'level': 1,
            'label': 'Red',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'ColorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Red color adjustments'
        },
        'ColorGreenSlider': {
            'level': 1,
            'label': 'Green',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'ColorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Green color adjustments'
        },
        'ColorBlueSlider': {
            'level': 1,
            'label': 'Blue',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'ColorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blue color adjustments'
        },
        'ColorBrightnessDecimalSlider': {
            'level': 1,
            'label': 'Brightness',
            'min_value': '0.00',
            'max_value': '2.00',
            'default': '1.00',
            'step': 0.01,
            'decimals': 2,
            'parentToggle': 'ColorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Changes the Brightness.'
        },
        'ColorContrastDecimalSlider': {
            'level': 1,
            'label': 'Contrast',
            'min_value': '0.00',
            'max_value': '2.00',
            'default': '1.00',
            'step': 0.01,
            'decimals': 2,
            'parentToggle': 'ColorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Changes the Contrast.'
        },
        'ColorSaturationDecimalSlider': {
            'level': 1,
            'label': 'Saturation',
            'min_value': '0.00',
            'max_value': '2.00',
            'default': '1.00',
            'step': 0.01,
            'decimals': 2,
            'parentToggle': 'ColorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Changes the Saturation.'
        },
        'ColorSharpnessDecimalSlider': {
            'level': 1,
            'label': 'Sharpness',
            'min_value': '0.0',
            'max_value': '2.0',
            'default': '1.0',
            'step': 0.1,
            'decimals': 1,
            'parentToggle': 'ColorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Changes the Sharpness.'
        },
        'ColorHueDecimalSlider': {
            'level': 1,
            'label': 'Hue',
            'min_value': '-0.50',
            'max_value': '0.50',
            'default': '0.00',
            'step': 0.01,
            'decimals': 2,
            'parentToggle': 'ColorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Changes the Hue.'
        },
        'ColorGammaDecimalSlider': {
            'level': 1,
            'label': 'Gamma',
            'min_value': '0.00',
            'max_value': '2.00',
            'default': '1.00',
            'step': 0.01,
            'decimals': 2,
            'parentToggle': 'ColorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Changes the Gamma.'
        },
        'ColorNoiseDecimalSlider': {
            'level': 1,
            'label': 'Noise',
            'min_value': '0.0',
            'max_value': '20.0',
            'default': '0.0',
            'step': 0.5,
            'decimals': 1,
            'parentToggle': 'ColorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Add noise to swapped face.'
        },

        'JPEGCompressionEnableToggle': {
            'level': 1,
            'label': 'JPEG Compression',
            'default': False,
            'help': 'Apply JPEG Compression to the swapped face to make output more realistic',
        },
        'JPEGCompressionAmountSlider': {
            'level': 2,
            'label': 'Compression',
            'min_value': '1',
            'max_value': '100',
            'default': '50',
            'step': 1,
            'parentToggle': 'JPEGCompressionEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the JPEG Compression amount'
        }
    },
    'Blend Adjustments':{
        'FinalBlendAdjEnableToggle': {
            'level': 1,
            'label': 'Final Blend',
            'default': False,
            'help': 'Blend at the end of pipeline.'
        },
        'FinalBlendAmountSlider': {
            'level': 2,
            'label': 'Final Blend Amount',
            'min_value': '1',
            'max_value': '50',
            'default': '1',
            'step': 1,
            'parentToggle': 'FinalBlendAdjEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the final blend value.'
        },
        'AdaptiveFeatherEnableToggle': {
            'level': 1,
            'label': 'Adaptive Feather',
            'default': False,
            'help': 'Feather proportionnel à la taille du visage pour des bords plus naturels.'
        },
        'AdaptiveFeatherPercentSlider': {
            'level': 2,
            'label': 'Feather (%)',
            'min_value': '0',
            'max_value': '100',
            'default': '15',
            'step': 1,
            'parentToggle': 'AdaptiveFeatherEnableToggle',
            'requiredToggleValue': True,
            'help': 'Pourcentage du plus grand côté du visage utilisé comme rayon de feather.'
        },
        'EdgeSmoothingEnableToggle': {
            'level': 1,
            'label': 'Edge Smoothing',
            'default': False,
            'help': 'Lissage guidé du bord du masque (guided/bilateral) pour réduire les halos.'
        },
        'EdgeSmoothingStrengthSlider': {
            'level': 2,
            'label': 'Smoothing Strength',
            'min_value': '0',
            'max_value': '100',
            'default': '30',
            'step': 5,
            'parentToggle': 'EdgeSmoothingEnableToggle',
            'requiredToggleValue': True,
            'help': 'Force du lissage sur la zone de bord (0 = désactivé).'
        },
        'WarpPresetSelection': {
            'level': 1,
            'label': 'Presets',
            'options': ['Manual', 'Steady Shot', 'Mobile Vlog', 'Portrait Studio'],
            'default': 'Manual',
            'help': 'Apply one-click profiles that set Warp Detail, Regularization, and Edge Smoothing.',
            'exec_function': common_widget_actions.apply_warp_preset,
            'exec_function_args': []
        },
        'OverallMaskBlendAmountSlider': {
            'level': 1,
            'label': 'Overall Mask Blend Amount',
            'min_value': '0',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'help': 'Combined masks blending distance. It is not applied to the border masks.'
        },
        'NonRigidWarpEnableToggle': {
            'level': 1,
            'label': 'Non-Rigid Warp (3D-like)',
            'default': False,
            'help': "Warp the swapped face and mask with dense landmarks (piecewise affine) for more natural local shape. Falls back to classic paste when insufficient points."
        },
        'NonRigidWarpModeSelection': {
            'level': 2,
            'label': 'Mode',
            'options': ['Manual', 'Auto'],
            'default': 'Auto',
            'parentToggle': 'NonRigidWarpEnableToggle',
            'requiredToggleValue': True,
            'help': 'Manual: always use when enabled. Auto: engage only when enough reliable landmarks and motion is low.'
        },        
        'NonRigidWarpDetailSlider': {
            'level': 2,
            'label': 'Warp Detail',
            'min_value': '1',
            'max_value': '100',
            'default': '60',
            'step': 5,
            'parentToggle': 'NonRigidWarpEnableToggle',
            'requiredToggleValue': True,
            'help': 'Controls deformation granularity. Lower = fewer control points (smoother, more rigid). Higher = more points (more local detail).'
        },
        'NonRigidWarpRegularizationSlider': {
            'level': 2,
            'label': 'Regularization (blend %)',
            'min_value': '0',
            'max_value': '100',
            'default': '20',
            'step': 5,
            'parentToggle': 'NonRigidWarpEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blend a portion of classic affine paste-back to stiffen the non-rigid warp (higher = more stable, less deformable).'
        },
    },
}