import tensorrt as trt
 
onnx_file_name = 'best.onnx'
tensorrt_file_name = 'yolov8n.engine'
fp_16_mode = True
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
 
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(EXPLICIT_BATCH)
parser = trt.OnnxParser(network, TRT_LOGGER)

config = builder.create_builder_config()

if fp_16_mode:
    config.set_flag(trt.BuilderFlag.FP16)  # FP16 모드 활성화
 
with open(onnx_file_name, 'rb') as model:
    if not parser.parse(model.read()):
        for error in range(parser.num_errors):
            print (parser.get_error(error))
 
engine = builder.build_serialized_network(network, config)

if engine is None:
	print("None")
with open(tensorrt_file_name, 'wb') as f:
    f.write(engine)
