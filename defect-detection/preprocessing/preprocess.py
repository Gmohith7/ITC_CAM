import cv2
import numpy as np


def draw_result(
    frame: np.ndarray,
    label: str,
    confidence: float,
    defect: bool = False,
    scanning: bool = False,
    regions: list = None,
    ocr_text: str = "",
) -> np.ndarray:
    """
    Overlay detection result on the frame (RGB in, RGB out).
    Amber border  = OCR not yet run (SCANNING).
    Green border  = batch code present (OK).
    Red border    = batch code missing (DEFECT).
    Blue boxes    = detected label regions.
    """
    h, w = frame.shape[:2]

    if scanning:
        border_color = (255, 180, 0)   # amber
        text_color   = (255, 180, 0)
        status       = "SCANNING..."
        conf_str     = ""
    elif defect:
        border_color = (255, 0, 0)     # red
        text_color   = (255, 0, 0)
        status       = "NO BATCH CODE"
        conf_str     = f"  {confidence:.0%}"
    else:
        border_color = (0, 200, 0)     # green
        text_color   = (0, 200, 0)
        status       = "BATCH CODE OK"
        conf_str     = f"  {confidence:.0%}"

    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), border_color, 6)

    cv2.putText(frame, status + conf_str, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, text_color, 3, cv2.LINE_AA)

    if ocr_text and not scanning:
        first_line = ocr_text.split('\n')[0][:60]
        cv2.putText(frame, first_line, (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)

    if regions and not scanning:
        region_color = (0, 120, 255)
        for (x, y, rw, rh) in regions:
            cv2.rectangle(frame, (x, y), (x + rw, y + rh), region_color, 2)

    return frame
