import cv2
import torch
from models import TRTModule  # YOLOv8-TensorRT 레포의 models.py
from config import CLASSES_DET, COLORS
from models.torch_utils import det_postprocess
from models.utils import blob, letterbox

ENGINE_PATH = "yolov8n_640_NMS.engine"  # 변환된 엔진 파일 경로
DEVICE = "cuda:0"  # Jetson Nano는 대부분 'cuda:0'
CAM_ID = 0  # 웹캠 기본 번호

def main():
    device = torch.device(DEVICE)
    Engine = TRTModule(ENGINE_PATH, device)
    H, W = Engine.inp_info[0].shape[-2:]  # 예: 320, 320

    Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])

    cap = cv2.VideoCapture(CAM_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image from webcam.")
            break

        draw = frame.copy()
        img, ratio, dwdh = letterbox(frame, (W, H))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = blob(rgb, return_seg=False)
        dwdh = torch.tensor(dwdh * 2, dtype=torch.float32, device=device)
        tensor = torch.tensor(tensor, device=device)

        # 추론
        data = Engine(tensor)
        bboxes, scores, labels = det_postprocess(data)

        if bboxes.numel() > 0:
            bboxes -= dwdh
            bboxes /= ratio

            for (bbox, score, label) in zip(bboxes, scores, labels):
                bbox = bbox.round().int().tolist()
                cls_id = int(label)
                cls = CLASSES_DET[cls_id]
                color = COLORS[cls]
                text = f'{cls}:{score:.3f}'
                x1, y1, x2, y2 = bbox

                (_w, _h), _bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)
                _y1 = min(y1 + 1, draw.shape[0])

                cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(draw, (x1, _y1), (x1 + _w, _y1 + _h + _bl), (0, 0, 255), -1)
                cv2.putText(draw, text, (x1, _y1 + _h), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        cv2.imshow('YOLOv8n-TRT-Webcam', draw)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
