import cv2
import torch
import torch.nn.functional as nnf
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor, Normalize

from face_mask_detection.face_detector import FaceDetector
from face_mask_detection.face_mask_classifier import Model


class FaceMaskDetector:
    def __init__(self):
        self.face_detector = FaceDetector(
            prototype='./models/deploy.prototxt.txt',
            model='./models/res10_300x300_ssd_iter_140000.caffemodel',
        )
        print('[INFO] Loaded face detector model')

        self.model = Model()
        self.model.load_state_dict(torch.load('./models/face_mask_detection_model.ckpt')['state_dict'], strict=False)
        print('[INFO] Loaded face mask classifier')

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("[INFO] Processing face mask detection on " + str(self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transformations = Compose([
            ToPILImage(),
            Resize((100, 100)),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.labels = ['No mask', 'Mask']
        self.label_color = [(10, 0, 255), (10, 255, 0)]

    def detect(self, frame):
        faces_coord = self.face_detector.detect(frame)
        for coord in faces_coord:
            start_x, start_y, width, height = coord
            start_x, start_y = max(start_x, 0), max(start_y, 0)

            face = frame[start_y:start_y + height, start_x:start_x + width]
            output = self.model(self.transformations(face).unsqueeze(0).to(self.device))
            prob = nnf.softmax(output, dim=1)
            top_p, top_class = prob.topk(1, dim=1)

            # draw face frame
            cv2.rectangle(frame,
                          (start_x, start_y),
                          (start_x + width, start_y + height),
                          (126, 65, 64),
                          thickness=2)

            text = self.labels[top_class] + " ({:.2f}%)".format(top_p.data.tolist()[0][0] * 100)

            text_size = cv2.getTextSize(text, self.font, 0.5, 2)[0]
            text_x = start_x + width // 2 - text_size[0] // 2

            # draw prediction label
            cv2.putText(frame,
                        text,
                        (text_x, start_y - 20),
                        self.font, 0.5, self.label_color[top_class], 2)
