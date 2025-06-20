from models import TRTModule  # isort:skip
import argparse
from pathlib import Path
import cv2
import torch
import time

from config import CLASSES_DET, COLORS
from models.torch_utils import det_postprocess
from models.utils import blob, letterbox, path_to_list

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, help='Engine file')
    parser.add_argument('--imgs', type=str, help='Images file')
    parser.add_argument('--show',
                        action='store_true',
                        help='Show the detection results')
    parser.add_argument('--out-dir',
                        type=str,
                        default='./output',
                        help='Path to output file')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='TensorRT infer device')
    args = parser.parse_args()
    return args

def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    Engine = TRTModule(args.engine, device)
    H, W = Engine.inp_info[0].shape[-2:]

    # set desired output names order
    Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])

    images = path_to_list(args.imgs)
    save_path = Path(args.out_dir)

    if not args.show and not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    # --- 동일한 입력 tensor 준비 (첫 번째 이미지로) ---
    image = images[0]
    bgr = cv2.imread(str(image))
    bgr, ratio, dwdh = letterbox(bgr, (W, H))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = blob(rgb, return_seg=False)
    dwdh = torch.tensor(dwdh * 2, dtype=torch.float32, device=device)
    tensor = torch.tensor(tensor, device=device)

    # --- 반복 추론 및 FPS 측정 ---
    num_repeats = 100  # 원하는 반복 횟수로 조절
    # warmup (캐시 등 영향 제거)
    for _ in range(10):
        _ = Engine(tensor)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_repeats):
        data = Engine(tensor)
        torch.cuda.synchronize()
        print(f"[{_+1}/{num_repeats}], {time.time()}")
    torch.cuda.synchronize()
    end = time.time()

    total_time = end - start
    fps = num_repeats / total_time
    

    # 결과 예시 출력 (옵션)
    bboxes, scores, labels = det_postprocess(data)
    print("Sample output:", bboxes, scores, labels)
    print(f"Repeated {num_repeats} times on same input tensor")
    print(f"Total inference time: {total_time:.4f} seconds")
    print(f"FPS: {fps:.2f}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
