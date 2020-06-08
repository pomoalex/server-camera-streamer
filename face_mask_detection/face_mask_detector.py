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

        self.pre_process_frame = Compose([
            ToPILImage(),
            Resize((100, 100)),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.labels = ['Without mask', 'With mask']
        self.label_color = [(10, 0, 255), (10, 255, 0)]

    def get_annotated_frame(self, frame):
        faces = self.face_detector.get_annotated_faces(frame)
        for face, coord in faces:
            output = self.model(self.pre_process_frame(face).unsqueeze(0).to(self.device))
            prediction = nnf.softmax(output, dim=1)
            self.annotate_frame(frame, coord, prediction)

    def annotate_frame(self, frame, coord, prediction):
        start_x, start_y, width, _ = coord
        top_p, top_class = prediction.topk(1, dim=1)

        text = self.labels[top_class] + " ({:.2f}%)".format(top_p.data[0, 0] * 100)
        text_size = cv2.getTextSize(text, self.font, 0.5, 2)[0]
        text_x = start_x + width // 2 - text_size[0] // 2

        cv2.putText(frame,
                    text,
                    (text_x, start_y - 20),
                    self.font, 0.5, self.label_color[top_class], 2)
