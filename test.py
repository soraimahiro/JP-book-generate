import cv2
import json

img = 'results/images/book0.jpg'
label = 'results/labels/book0.json'

img = cv2.imread(img)
with open(label, 'r', encoding='utf-8') as f:
    annotations = json.load(f)
    for ann in annotations:
        bbox = ann['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()