"""Generate assets/watermark.png (transparent PNG). Replace it with your own logo any time."""
import sys
from pathlib import Path

import cv2
import numpy as np

text = sys.argv[1] if len(sys.argv) > 1 else "HighlightHQ"
out = Path(__file__).resolve().parent.parent / "assets" / "watermark.png"
font, scale, thick = cv2.FONT_HERSHEY_DUPLEX, 2.2, 4
(tw, th), base = cv2.getTextSize(text, font, scale, thick)
pad = 24
img = np.zeros((th + base + 2 * pad, tw + 2 * pad, 4), np.uint8)
# rounded dark pill background
cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (20, 20, 20, 150), -1)
org = (pad, pad + th)
cv2.putText(img, text, org, font, scale, (0, 0, 0, 255), thick + 6, cv2.LINE_AA)
cv2.putText(img, text, org, font, scale, (0, 200, 255, 255), thick, cv2.LINE_AA)
out.parent.mkdir(exist_ok=True)
cv2.imwrite(str(out), img)
print(out)
