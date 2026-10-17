import cv2
import numpy as np


def estimate_fruit_maturity(color_bgr, mask):
    pixels = color_bgr[mask.astype(bool)]
    if pixels.size == 0:
        return {
            'maturity': 'unknown',
            'ripe_score': 0.0,
            'red_ratio': 0.0,
            'orange_ratio': 0.0,
            'green_ratio': 0.0,
        }

    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    h = hsv[:, 0]
    s = hsv[:, 1]
    v = hsv[:, 2]
    valid = (s > 45) & (v > 35)
    if not valid.any():
        return {
            'maturity': 'unknown',
            'ripe_score': 0.0,
            'red_ratio': 0.0,
            'orange_ratio': 0.0,
            'green_ratio': 0.0,
        }

    hv = h[valid]
    red = ((hv <= 10) | (hv >= 170)).sum()
    orange = ((hv > 10) & (hv <= 28)).sum()
    green = ((hv >= 35) & (hv <= 90)).sum()
    total = float(len(hv))
    red_ratio = float(red / total)
    orange_ratio = float(orange / total)
    green_ratio = float(green / total)
    ripe_score = red_ratio + 0.5 * orange_ratio

    if red_ratio > 0.55:
        maturity = 'ripe'
    elif green_ratio > 0.45:
        maturity = 'unripe'
    else:
        maturity = 'semi_ripe'

    return {
        'maturity': maturity,
        'ripe_score': float(ripe_score),
        'red_ratio': red_ratio,
        'orange_ratio': orange_ratio,
        'green_ratio': green_ratio,
    }
