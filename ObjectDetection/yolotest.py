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
		image = input_image.transpose(0,3,1,2)
		image = np.ascontiguousarray(image)
		cuda.memcpy_htod(self.inputs[0]['allocation'], image)
		self.context.execute_v2(self.allocations)
		for o in range(len(self.outputs)):
			cuda.memcpy_dtoh(self.outputs[o]['host_allocation'], self.outputs[o]['allocation'])

		result = self.outputs[0]['host_allocation']
		return result
	
def extract_boxes_and_classes(output_tensor, conf_threshold=0.5):
	# Extract the bounding box, objectness, and class probabilities
	bbox = output_tensor[:4]  # [x_center, y_center, width, height]
	class_probs = output_tensor[4:]  # Class probabilities

	score = np.max(class_probs)
	target = np.argmax(class_probs)
	class_id = 0 if target < 8400 else 1

	if score >= conf_threshold:
		# (center_x, center_y, width, height)
		box = bbox[:,target % 8400]

	return box, class_id, score

if __name__ == "__main__":
	engine_path = "ob_detection.engine"
	nn_engine = Engine(engine_path)

	for imgName in os.listdir("./images")[:10]:
		img = [cv2.imread(f'./images/{imgName}')]
		input_image = np.array(img, dtype=np.float32)
		input_image /= 255.0
		
		start = time.time()
		results = nn_engine.infer(input_image)
		end = time.time()
		infer_time = end - start

		print(f'image name : {imgName}')
		print(f'infer_time : {infer_time:.5f} sec')

		box, class_id, score = extract_boxes_and_classes(results[0])
		print(f"Class ID: {class_id}, Confidence Score: {score:.2f}, Bounding Box: {box}")