from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Tuple

from PySide6 import QtWidgets

from app.ui.widgets.actions import common_actions as common_widget_actions

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow


# Minimal, safe defaults for an "Ultra Realism" trio of presets
def _webcam_preset() -> Tuple[Dict, Dict]:
    control_updates: Dict = {}
    param_updates: Dict = {
        'SwapModelSelection': 'InStyleSwapper256 Version B',
        'StrengthEnableToggle': True,
        'StrengthAmountSlider': 100,
        'FaceParserEnableToggle': True,
        'DFLXSegEnableToggle': True,
        'DFLXSegSizeSlider': 10,
        'OccluderXSegBlurSlider': 3,
        'BorderTopSlider': 4,
        'BorderLeftSlider': 4,
        'BorderRightSlider': 4,
        'BorderBottomSlider': 4,
        'BorderBlurSlider': 5,
        'OverallMaskBlendAmountSlider': 6,
        'RestoreMouthEnableToggle': True,
        'RestoreMouthBlendAmountSlider': 30,
        'RestoreMouthFeatherBlendSlider': 4,
        'RestoreEyesEnableToggle': True,
        'RestoreEyesBlendAmountSlider': 15,
        'RestoreEyesFeatherBlendSlider': 4,
    'AutoColorEnableToggle': True,
    'AutoColorTransferTypeSelection': 'DFL_Test',
        'AutoColorBlendAmountSlider': 45,
        'DifferencingEnableToggle': True,
        'DifferencingAmountSlider': 15,
        'DifferencingBlendAmountSlider': 3,
        'FinalBlendAdjEnableToggle': True,
        'FinalBlendAmountSlider': 1,
        'JPEGCompressionEnableToggle': True,
        'ColorNoiseDecimalSlider': 0.04,
        'SimilarityThresholdSlider': 70,
    }
    return control_updates, param_updates


def _cinema_preset() -> Tuple[Dict, Dict]:
    control_updates: Dict = {}
    param_updates: Dict = {
        'SwapModelSelection': 'GhostFace-v3',  # or 'SimSwap512' depending on preference
        'StrengthEnableToggle': True,
        'StrengthAmountSlider': 90,
        'FaceParserEnableToggle': True,
        'DFLXSegEnableToggle': True,
        'DFLXSegSizeSlider': 8,
        'OccluderXSegBlurSlider': 3,
        'BorderTopSlider': 3,
        'BorderLeftSlider': 3,
        'BorderRightSlider': 3,
        'BorderBottomSlider': 3,
        'BorderBlurSlider': 4,
        'OverallMaskBlendAmountSlider': 6,
        'RestoreMouthEnableToggle': True,
        'RestoreMouthBlendAmountSlider': 25,
        'RestoreMouthFeatherBlendSlider': 4,
        'RestoreEyesEnableToggle': True,
        'RestoreEyesBlendAmountSlider': 12,
        'RestoreEyesFeatherBlendSlider': 4,
        'AutoColorEnableToggle': True,
        'AutoColorTransferTypeSelection': 'Test_Mask',
        'AutoColorBlendAmountSlider': 40,
        'DifferencingEnableToggle': True,
        'DifferencingAmountSlider': 10,
        'DifferencingBlendAmountSlider': 3,
        'FinalBlendAdjEnableToggle': True,
        'FinalBlendAmountSlider': 2,
        'JPEGCompressionEnableToggle': True,
        'ColorNoiseDecimalSlider': 0.03,
        'ColorGammaDecimalSlider': 0.98,
        'SimilarityThresholdSlider': 70,
    }
    return control_updates, param_updates


def _action_preset() -> Tuple[Dict, Dict]:
    control_updates: Dict = {}
    param_updates: Dict = {
        'SwapModelSelection': 'InStyleSwapper256 Version C',
        'StrengthEnableToggle': True,
        'StrengthAmountSlider': 95,
        'FaceParserEnableToggle': True,
        'DFLXSegEnableToggle': True,
        'DFLXSegSizeSlider': 12,
        'OccluderXSegBlurSlider': 4,
        'BorderTopSlider': 5,
        'BorderLeftSlider': 5,
        'BorderRightSlider': 5,
        'BorderBottomSlider': 5,
        'BorderBlurSlider': 6,
        'OverallMaskBlendAmountSlider': 8,
        'RestoreMouthEnableToggle': True,
        'RestoreMouthBlendAmountSlider': 28,
        'RestoreMouthFeatherBlendSlider': 5,
        'RestoreEyesEnableToggle': True,
        'RestoreEyesBlendAmountSlider': 14,
        'RestoreEyesFeatherBlendSlider': 5,
    'AutoColorEnableToggle': True,
    'AutoColorTransferTypeSelection': 'DFL_Test',
        'AutoColorBlendAmountSlider': 35,
        'DifferencingEnableToggle': True,
        'DifferencingAmountSlider': 12,
        'DifferencingBlendAmountSlider': 3,
        'FinalBlendAdjEnableToggle': True,
        'FinalBlendAmountSlider': 1,
        'JPEGCompressionEnableToggle': True,
        'ColorNoiseDecimalSlider': 0.03,
        'SimilarityThresholdSlider': 72,
    }
    return control_updates, param_updates


PRESETS: Dict[str, Tuple] = {
    'Webcam': _webcam_preset,
    'Cinéma': _cinema_preset,
    'Action': _action_preset,
}

# --- New profiles: Portrait Stable, Vlog Mobile, Low-Light ---

def _portrait_stable_preset() -> Tuple[Dict, Dict]:
    """Indoor/studio, low motion. Conservative AI assist, gentle stabilization.
    """
    control_updates: Dict = {
        # AI Smart Tuning (transient adjustments)
        'AISmartTuningEnableToggle': True,
        'AISmartTuningModeSelection': 'Conservative',
        'AISmartTuningReactivitySlider': 24,
        # Auto Threshold Assist (recognition)
        'AutoThresholdAssistEnableToggle': True,
        'AutoThresholdAssistMaxDropSlider': 6,
        'AutoThresholdAssistFramesPerStepSlider': 20,
        # Rotation stabilization
        'RotationStabilizationEnableToggle': True,
        'RotationStabilizationPresetSelection': 'Medium',
        'RotationRollThresholdSlider': 3,
    }
    param_updates: Dict = {}
    return control_updates, param_updates


def _vlog_mobile_preset() -> Tuple[Dict, Dict]:
    """Handheld vlog, high motion. Aggressive AI assist, strong stabilization.
    """
    control_updates: Dict = {
        'AISmartTuningEnableToggle': True,
        'AISmartTuningModeSelection': 'Aggressive',
        'AISmartTuningReactivitySlider': 8,
        'AutoThresholdAssistEnableToggle': True,
        'AutoThresholdAssistMaxDropSlider': 12,
        'AutoThresholdAssistFramesPerStepSlider': 10,
        'RotationStabilizationEnableToggle': True,
        'RotationStabilizationPresetSelection': 'High',
        'RotationRollThresholdSlider': 4,
    }
    param_updates: Dict = {}
    return control_updates, param_updates


def _low_light_preset() -> Tuple[Dict, Dict]:
    """Low illumination scenes. Balanced/aggressive threshold help, medium stabilization.
    """
    control_updates: Dict = {
        'AISmartTuningEnableToggle': True,
        'AISmartTuningModeSelection': 'Balanced',
        'AISmartTuningReactivitySlider': 12,
        'AutoThresholdAssistEnableToggle': True,
        'AutoThresholdAssistMaxDropSlider': 15,
        'AutoThresholdAssistFramesPerStepSlider': 12,
        'RotationStabilizationEnableToggle': True,
        'RotationStabilizationPresetSelection': 'Medium',
        'RotationRollThresholdSlider': 3,
    }
    param_updates: Dict = {}
    return control_updates, param_updates


# Extend the presets registry with the new profiles
PRESETS.update({
    'Portrait Stable': _portrait_stable_preset,
    'Vlog Mobile': _vlog_mobile_preset,
    'Low-Light': _low_light_preset,
})


def apply_preset(main_window: 'MainWindow', preset_name: str):
    if preset_name not in PRESETS:
        common_widget_actions.create_and_show_toast_message(
            main_window, 'Preset', f'Preset "{preset_name}" non trouvé', style_type='warning')
        return

    # Persist the last used preset in control so workspace saves it
    if 'PresetSelection' not in main_window.control:
        common_widget_actions.create_control(main_window, 'PresetSelection', preset_name)
    else:
        main_window.control['PresetSelection'] = preset_name

    control_updates, param_updates = PRESETS[preset_name]()

    # Apply control updates
    for k, v in control_updates.items():
        common_widget_actions.update_control(main_window, k, v)

    # Apply parameters to all existing target faces
    if main_window.target_faces:
        prev_sel = main_window.selected_target_face_id
        for face_id in list(main_window.target_faces.keys()):
            # Temporarily switch selection to update this face's parameters
            main_window.selected_target_face_id = face_id
            for pk, pv in param_updates.items():
                common_widget_actions.update_parameter(main_window, pk, pv, enable_refresh_frame=False)
        # Restore previous selection
        main_window.selected_target_face_id = prev_sel
    else:
        # No faces yet: prime current widget parameters so future faces inherit
        for pk, pv in param_updates.items():
            main_window.current_widget_parameters[pk] = pv

    # Also update the currently selected face UI if one is active
    sel_id = main_window.selected_target_face_id
    if sel_id and sel_id in main_window.parameters:
        # Ensure UI reflects parameter changes for selected face
        from app.ui.widgets.actions.common_actions import set_widgets_values_using_face_id_parameters
        set_widgets_values_using_face_id_parameters(main_window, sel_id)

    # Feedback and refresh
    common_widget_actions.create_and_show_toast_message(
        main_window, 'Preset appliqué', f'Ultra Realism · {preset_name}', style_type='success')
    common_widget_actions.refresh_frame(main_window)

    # Update split-button tooltip if present
    try:
        if hasattr(main_window, 'ultraRealismButton') and main_window.ultraRealismButton:
            main_window.ultraRealismButton.setToolTip(
                f"Preset: {main_window.control.get('PresetSelection', preset_name)} (click to re-apply)"
            )
    except Exception:
        pass
