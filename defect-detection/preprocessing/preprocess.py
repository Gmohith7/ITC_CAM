import cv2
import numpy as np
import config


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Resize to INFERENCE_SIZE, normalise to [0, 1], add batch dim.
    Input:  (H, W, 3) uint8 RGB
    Output: (1, H, W, 3) float32
    """
    resized = cv2.resize(frame, config.INFERENCE_SIZE)
    normalised = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalised, axis=0)


def draw_result(frame: np.ndarray, label: str, confidence: float, defect: bool = False) -> np.ndarray:
    """Overlay prediction label and confidence onto the frame (in-place copy)."""
    color = (0, 0, 255) if defect else (0, 255, 0)
    text = f"{label}: {confidence:.2f}"
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
    border_color = (0, 0, 255) if defect else (0, 255, 0)
    cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), border_color, 4)
    return frame
