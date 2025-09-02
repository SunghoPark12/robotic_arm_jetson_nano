from models import TRTModule
import torch
import numpy as np
import argparse

def main(args):
    device = torch.device(args.device)
    Engine = TRTModule(args.engine, device)
    H, W = Engine.inp_info[0].shape[-2:]  # 입력 이미지 크기 확인

    # .npy 파일에서 입력 데이터 로드 (float32, [1, 3, H, W] 형태 권장)
    input_data = np.load(args.input_npy)
    print(f"Loaded input shape: {input_data.shape}, dtype: {input_data.dtype}")

    # torch tensor로 변환 및 디바이스 이동
    tensor = torch.as_tensor(input_data, dtype=torch.float32, device=device)

    # 추론 실행
    print("Running inference...")
    data = Engine(tensor)

    # tuple 출력 해석
    if isinstance(data, tuple):
        if len(data) == 4:
            num_dets, bboxes, scores, labels = data
            print("num_dets:", num_dets)
            print("bboxes:", bboxes)
            print("scores:", scores)
            print("labels:", labels)
        else:
            print(f"Unexpected tuple length: {len(data)}")
            for i, v in enumerate(data):
                print(f"Output {i}: {v}")
    else:
        print("Output type is not tuple. Type:", type(data))
        print(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, required=True, help='Engine file (.engine)')
    parser.add_argument('--input-npy', type=str, required=True, help='Input .npy file')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    args = parser.parse_args()

    main(args)
