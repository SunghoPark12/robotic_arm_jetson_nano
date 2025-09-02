import numpy as np
import onnxruntime as ort
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# 1. 입력 데이터 생성 (320x320, float32)
input_shape = (1, 3, 320, 320)  # 예시: batch=1, C=3, H=320, W=320
dummy_input = np.random.rand(*input_shape).astype(np.float32)

# 2. ONNX 추론
onnx_session = ort.InferenceSession("yolov8n_320_NMS.onnx", providers=['CPUExecutionProvider'])
input_name = onnx_session.get_inputs()[0].name
onnx_outputs = onnx_session.run(None, {input_name: dummy_input})

print("ONNX output shapes:", [o.shape for o in onnx_outputs])

# 3. TensorRT 추론 함수
def infer_trt(engine_path, input_data):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        # Prepare buffers
        inputs = []
        outputs = []
        bindings = []

        for binding in engine:
            binding_shape = context.get_binding_shape(binding)
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            size = int(np.prod(binding_shape))

            if engine.binding_is_input(binding):
                input_arr = np.ascontiguousarray(input_data.reshape(binding_shape)).astype(dtype)
                input_mem = cuda.mem_alloc(input_arr.nbytes)
                cuda.memcpy_htod(input_mem, input_arr)
                inputs.append(input_mem)
                bindings.append(int(input_mem))
            else:
                output_arr = np.empty(binding_shape, dtype=dtype)
                output_mem = cuda.mem_alloc(output_arr.nbytes)
                outputs.append((output_arr, output_mem))
                bindings.append(int(output_mem))

        # Run inference
        context.execute_v2(bindings)

        output_arrays = []
        for arr, mem in outputs:
            cuda.memcpy_dtoh(arr, mem)
            output_arrays.append(arr.copy())
        return output_arrays

# 4. TensorRT 추론
trt_outputs = infer_trt("yolov8n_320_NMS.engine", dummy_input)
print("TRT output shapes:", [o.shape for o in trt_outputs])

# 5. 결과 비교 (각 output별로)
for i, (onnx_out, trt_out) in enumerate(zip(onnx_outputs, trt_outputs)):
    # 일부 int 타입 출력은 float으로 캐스팅 필요
    if onnx_out.dtype != trt_out.dtype:
        onnx_out = onnx_out.astype(trt_out.dtype)
    diff = np.abs(onnx_out - trt_out)
    print(f"Output {i}: max abs diff={diff.max()}, mean abs diff={diff.mean()}")
    print(f"ONNX min/max: {onnx_out.min()}/{onnx_out.max()}, TRT min/max: {trt_out.min()}/{trt_out.max()}")
