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

    While scanning:  thin neutral grey border only — no text, no boxes.
    BATCH CODE OK:   green pill badge bottom-left + highlighted region box.
    DEFECT:          red pill badge bottom-left only (no box — nothing found).
    """
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

    h, w = frame.shape[:2]

    # ── Scanning: minimal UI — just a thin border so the user knows it's running ──
    if scanning:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (100, 100, 100), 2)
        _draw_pill(frame, "SCANNING...", (160, 160, 160), w, h)
        return frame

    # ── Draw detected region boxes first (behind the badge) ──────────────────
    if not defect and regions:
        for (x, y, rw, rh) in regions:
            # Bright cyan box around the confirmed batch code region
            cv2.rectangle(frame, (x, y), (x + rw, y + rh), (0, 230, 180), 3)

    # ── Status badge (pill shape, bottom-left, always inside frame) ──────────
    if defect:
        badge_color = (220, 30, 30)
        badge_text  = f"NO BATCH CODE"
        _draw_pill(frame, badge_text, badge_color, w, h)
    else:
        badge_color = (20, 200, 80)
        badge_text  = f"BATCH CODE OK  {confidence:.0%}"
        _draw_pill(frame, badge_text, badge_color, w, h)

        # Show first line of OCR text as a small subtitle above the badge
        if ocr_text:
            first_line = ocr_text.split('\n')[0][:70].strip()
            if first_line:
                font_scale = max(0.4, min(0.65, w / 1200))
                thickness  = max(1, round(font_scale * 2))
                (tw, th), _ = cv2.getTextSize(first_line, cv2.FONT_HERSHEY_SIMPLEX,
                                              font_scale, thickness)
                x_txt = 24
                y_txt = h - 24 - 42 - th - 6   # sits just above the pill
                # dark backing for readability
                cv2.rectangle(frame,
                              (x_txt - 6, y_txt - th - 4),
                              (x_txt + tw + 6, y_txt + 6),
                              (0, 0, 0), -1)
                cv2.putText(frame, first_line,
                            (x_txt, y_txt),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            (220, 220, 220), thickness, cv2.LINE_AA)

    return frame


def _draw_pill(frame: np.ndarray, text: str, color: tuple, w: int, h: int):
    """
    Draw a filled rounded-rectangle badge at the bottom-left of the frame.
    Guaranteed to stay fully inside the image regardless of resolution.
    """
    font        = cv2.FONT_HERSHEY_SIMPLEX
    font_scale  = max(0.55, min(1.2, w / 900))
    thickness   = max(2, round(font_scale * 2.2))
    pad_x, pad_y = 20, 10

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pill_w = tw + pad_x * 2
    pill_h = th + pad_y * 2 + baseline

    margin = 20
    x1 = margin
    y1 = h - margin - pill_h
    x2 = x1 + pill_w
    y2 = y1 + pill_h

    # Clamp so it never goes out of bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    # Semi-transparent fill
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    # Outline
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Text — baseline-anchored inside the pill
    tx = x1 + pad_x
    ty = y1 + pad_y + th
    ty = min(ty, h - margin - baseline - 2)
    cv2.putText(frame, text, (tx, ty), font, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)
