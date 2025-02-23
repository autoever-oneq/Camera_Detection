import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time
import cv2
import os

class Engine:
	def __init__(self, engine_path):
		self.logger = trt.Logger(trt.Logger.WARNING)
		self.runtime = trt.Runtime(self.logger)

		self.engine = None
		self.context = None
		trt.init_libnvinfer_plugins(None, "")

		with open(engine_path, "rb") as f:
			serialized_engine = f.read()
			self.engine = self.runtime.deserialize_cuda_engine(serialized_engine)
			self.context = self.engine.create_execution_context()

		if self.engine is None or self.context is None:
			print("None..")
			exit()

		self.inputs, self.outputs, self.allocations = self.allocate_buffers()

	# class HostDeviceMem:
	# 	def __init__(self, host_mem, device_mem):
	# 		self.host = host_mem
	# 		self.device = device_mem

	
	def allocate_buffers(self):
		inputs, outputs, allocations = [], [], []

		# print(f"self.engine.numbindings={self.engine.num_bindings}, num_layers={self.engine.num_layers}")

		for i in range(self.engine.num_bindings):
			is_input = False
			if self.engine.binding_is_input(i):
				is_input = True
			
			name = self.engine.get_binding_name(i)
			dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(i)))
			shape = self.context.get_binding_shape(i)

			if is_input and shape[0] < 0:
				assert self.engine.num_optimization_profiles > 0
				profile_shape = self.engine.get_profile_shape(0, name)
				assert len(profile_shape) == 3
				self.context.set_binding_shape(i, profile_shape[2])
				shape = self.context.get_binding_shape(i)
			
			if is_input:
				self.batch_size = shape[0]
			size = dtype.itemsize
			for s in shape:
				size *= s

			# Allocate host and device buffers
			allocation = cuda.mem_alloc(size)
			host_allocation = None if is_input else np.zeros(shape, dtype)
			binding = {
				"index": i,
				"name": name,
				"dtype": dtype,
				"shape": list(shape),
				"allocation": allocation,
				"host_allocation": host_allocation,
			}

			# Append to the appropiate input/output list
			allocations.append(allocation)
			if self.engine.binding_is_input(i):  # 🔹 get_tensor_mode() 대신 사용
				inputs.append(binding)
			else:
				outputs.append(binding)
		return inputs, outputs, allocations

	def infer(self, input_image):
		# Transfer input data to device
		image = input_image.transpose(0,3,1,2)	# (B,H,W,C) -> (B,C,H,W)
		image = np.ascontiguousarray(image)
		cuda.memcpy_htod(self.inputs[0]['allocation'], image)
		self.context.execute_v2(self.allocations)
		for o in range(len(self.outputs)):
			cuda.memcpy_dtoh(self.outputs[o]['host_allocation'], self.outputs[o]['allocation'])

		result = self.outputs[0]['host_allocation']
		return result
	
	def extract_boxes_and_classes(self, output_tensor, conf_threshold=0.5):
		# classes_id, boxes, scores = [], [], []
		object_info = []

		# Extract the bounding box, objectness, and class probabilities
		bbox = output_tensor[:4]  # [x_center, y_center, width, height]
		probs = output_tensor[4:]	# (0th class probs[], 1st class probs[], ...)

		for cls, prob in enumerate(probs):
			target_col = np.argmax(prob)	# Index of the most likely bbox of each car

			print(f"class {cls} conf : {prob[target_col]}")
			if prob[target_col] >= conf_threshold:
				box = bbox[:,target_col]
				conf = prob[target_col]

				object_info.append((cls, box, conf))

		return object_info

if __name__ == "__main__":
	engine_path = "yolov8n.engine"
	nn_engine = Engine(engine_path)

	for imgName in os.listdir("./images")[:1]:
		# Load image & normalize to 0~1
		img = [cv2.imread(f'./images/{imgName}')]
		input_image = np.array(img, dtype=np.float32)
		input_image /= 255.0
		
		start = time.time()
		results = nn_engine.infer(input_image)
		end = time.time()
		infer_time = end - start

		print(f'image name : {imgName}')
		print(f'infer_time : {infer_time:.5f} sec')

		object_info = nn_engine.extract_boxes_and_classes(results[0])

		for info in object_info:
			print(f"(Class ID, BBox, Confidence): {info}")