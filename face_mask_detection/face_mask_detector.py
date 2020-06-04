import torch
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor

from face_mask_detection.face_detector import FaceDetector
from face_mask_detection.face_mask_classifier import Model


class FaceMaskDetector:
    def __init__(self):
        self.faceDetector = FaceDetector(
            prototype='./models/deploy.prototxt.txt',
            model='./models/res10_300x300_ssd_iter_140000.caffemodel',
        )
        print('[INFO] Loaded face detector model')

        self.model = Model()
        self.model.load_state_dict(torch.load('./models/face_mask.ckpt')['state_dict'], strict=False)
        print('[INFO] Loaded face mask classifier')

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("[INFO] Processing face mask detection on " + str(self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transformations = Compose([
            ToPILImage(),
            Resize((100, 100)),
            ToTensor(),
        ])
