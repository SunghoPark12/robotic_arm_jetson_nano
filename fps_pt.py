import time
from ultralytics import YOLO

# 모델 로드 (현재 디렉토리에 best.pt 있을 경우)
model = YOLO('best.pt')

# ---------------------
# Warm-up (10회)
print("Warm-up start...")
for _ in range(10):
    _ = model.predict(source='data/bus.jpg', show=False)
print("Warm-up done.")

# ---------------------
# 본 측정 (100회 반복)
print("Start FPS measurement...")
start = time.time()
results = None
for _ in range(100):
    results = model.predict(source='data/bus.jpg', show=False)
end = time.time()

# ---------------------
# 결과 출력
fps = 100 / (end - start)

print(f"\n=== 결과 출력 ===")


# 전체 results 객체 출력
print(f"\nFull results object:\n{results}")

# 결과 상세 출력 (박스 좌표, 클래스, 신뢰도 등)
for result in results:
    print(f"\n--- Detection Result ---")
    print(f"Boxes: {result.boxes.xyxy}")      # xyxy 좌표
    print(f"Classes: {result.boxes.cls}")     # 클래스
    print(f"Confidences: {result.boxes.conf}") # 신뢰도

print(f"YOLOv8 Custom Model (best.pt) FPS: {fps:.2f}")
