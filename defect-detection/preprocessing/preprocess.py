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

    Border colours:
      Amber — OCR not yet run (SCANNING)
      Green — batch code present (OK)
      Red   — batch code missing (DEFECT)
    Blue rectangles mark detected label regions.
    """
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

    h, w = frame.shape[:2]
    # Scale text and UI elements relative to a 640-wide reference frame.
    scale = max(0.5, w / 640)
    thickness = max(1, round(scale))
    border_t = max(3, round(scale * 6))

    if scanning:
        border_color = (255, 180, 0)
        text_color   = (255, 180, 0)
        status       = "SCANNING..."
        conf_str     = ""
    elif defect:
        border_color = (255, 0, 0)
        text_color   = (255, 0, 0)
        status       = "NO BATCH CODE"
        conf_str     = f"  {confidence:.0%}"
    else:
        border_color = (0, 200, 0)
        text_color   = (0, 200, 0)
        status       = "BATCH CODE OK"
        conf_str     = f"  {confidence:.0%}"

    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), border_color, border_t)

    font_large = scale * 1.4
    font_small = scale * 0.7
    y_status   = max(40, round(h * 0.05))
    y_text     = y_status + round(h * 0.06)

    cv2.putText(frame, status + conf_str, (20, y_status),
                cv2.FONT_HERSHEY_SIMPLEX, font_large, text_color,
                max(2, thickness * 2), cv2.LINE_AA)

    if ocr_text and not scanning:
        first_line = ocr_text.split('\n')[0][:80]
        cv2.putText(frame, first_line, (20, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, font_small, (220, 220, 220),
                    thickness + 1, cv2.LINE_AA)

    if regions and not scanning:
        for (x, y, rw, rh) in regions:
            cv2.rectangle(frame, (x, y), (x + rw, y + rh), (0, 120, 255), thickness + 1)

    return frame
