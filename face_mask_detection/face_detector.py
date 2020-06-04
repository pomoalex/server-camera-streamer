import numpy as np
from cv2 import resize
from cv2.dnn import blobFromImage, readNetFromCaffe


class FaceDetector:
    def __init__(self, prototype, model, confidence_threshold: float = 0.6):
        self.prototype = prototype
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.classifier = readNetFromCaffe(str(prototype), str(model))

    def detect(self, image):
        net = self.classifier
        height, width = image.shape[:2]
        blob = blobFromImage(resize(image, (300, 300)), 1.0,
                             (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        faces = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < self.confidence_threshold:
                continue
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            start_x, start_y, end_x, end_y = box.astype("int")
            faces.append(np.array([start_x, start_y, end_x - start_x, end_y - start_y]))
        return faces
