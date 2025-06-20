import cv2
import torch
import sys
import numpy as np
from models import TRTModule
from config import CLASSES_DET, COLORS
from models.torch_utils import det_postprocess
from models.utils import blob, letterbox

class Detector:
    def __init__(self, engine_path, device='cuda:0', conf_thresh=0.3):
        self.device = torch.device(device)
        self.Engine = TRTModule(engine_path, self.device)
        self.H, self.W = self.Engine.inp_info[0].shape[-2:]
        self.Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])
        self.conf_thresh = conf_thresh

    def detect_frame(self, frame):
        bgr, ratio, dwdh = letterbox(frame, (self.W, self.H))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = blob(rgb, return_seg=False)
        dwdh = torch.tensor(dwdh * 2, dtype=torch.float32, device=self.device)
        tensor = torch.tensor(tensor, device=self.device)
        data = self.Engine(tensor)
        bboxes, scores, labels = det_postprocess(data)
        results = []
        if bboxes.numel() > 0:
            bboxes -= dwdh
            bboxes /= ratio
            for (bbox, score, label) in zip(bboxes, scores, labels):
                bbox = bbox.round().int().tolist()
                cls_id = int(label)
                cls = CLASSES_DET[cls_id]
                results.append({
                    "class": cls,
                    "score": float(score),
                    "bbox": bbox
                })
        return results

    def detect_n_frames_with_overlay(self, cap, window_name, n_frames=8, delay=1):
       #n프레임 동안 객체인식한 결과를 전체 전달
        results_batch = []
        for i in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                break
            results = self.detect_frame(frame)
            # Overlay
            overlay = frame.copy()
            for det in results:
                x1, y1, x2, y2 = det["bbox"]
                cls = det["class"]
                color = COLORS[cls]
                text = f'{cls}:{det["score"]:.3f}'
                (_w, _h), _bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)
                _y1 = min(y1 + 1, overlay.shape[0])
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(overlay, (x1, _y1), (x1 + _w, _y1 + _h + _bl), (0, 0, 255), -1)
                cv2.putText(overlay, text, (x1, _y1 + _h), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.imshow(window_name, overlay)
            cv2.waitKey(1)
            results_batch.append(results)
        #overlay 잔상 제거
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)
        return results_batch

    @staticmethod
    def smooth_all_objects(results_batch, iou_thresh=0.65, method='mean'):
        # results_batch를 하나의 결과로 smoothing하여 반환
        tracklets = []
        for frame_results in results_batch:
            for det in frame_results:
                matched = False
                for track in tracklets:
                    if det['class'] == track[0]['class']:
                        ious = [Detector.bbox_iou(np.array(det['bbox']), np.array(hist['bbox'])) for hist in track]
                        if max(ious) > iou_thresh:
                            track.append(det)
                            matched = True
                            break
                if not matched:
                    tracklets.append([det])
        smoothed_objects = []
        for track in tracklets:
            bboxes = np.array([o['bbox'] for o in track])
            scores = np.array([o['score'] for o in track])
            if method == 'mean':
                bbox_smooth = np.mean(bboxes, axis=0)
                score_smooth = np.mean(scores)
            elif method == 'median':
                bbox_smooth = np.median(bboxes, axis=0)
                score_smooth = np.median(scores)
            else:
                bbox_smooth = bboxes[-1]
                score_smooth = scores[-1]
            if (len(track) >= 3): #admit object detected over 3 frames only
                smoothed_objects.append({
                    'class': track[0]['class'],
                    'bbox': bbox_smooth.tolist(),
                    'score': float(score_smooth),
                    'count': len(track)
                })
        return smoothed_objects

    @staticmethod
    def bbox_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
        area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0


    def obj_pixel(self, cap, window_name, n_frames=8):
        for _ in range(3):
            results_batch = self.detect_n_frames_with_overlay(cap, window_name, n_frames)
            smoothed = self.smooth_all_objects(results_batch)
            if len(smoothed) != 0:
                break
        if len(smoothed) == 0:
            print("no object detected")
            sys.exit(0)
        else:
            print("smoothing result:", smoothed)
            first_obj=max(smoothed, key=lambda x: x['score'])
            cls = first_obj['class']
            x_center = (first_obj['bbox'][0] + first_obj['bbox'][2]) / 2
            y_center = (first_obj['bbox'][1] + first_obj['bbox'][3]) / 2
            return cls, x_center, y_center





