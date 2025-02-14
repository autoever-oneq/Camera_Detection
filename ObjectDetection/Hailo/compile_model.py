from hailo_sdk_client import ClientRunner

# Define the quantized model HAR file
model_name = "yolov8n"
quantized_model_har_path = f"{model_name}_quantized_model.har"

# Initialize the ClientRunner with the HAR file
runner = ClientRunner(har=quantized_model_har_path)
print("[info] ClientRunner initialized successfully.")

# Compile the model
try:
    hef = runner.compile()
    print("[info] Compilation completed successfully.")
except Exception as e:
    print(f"[error] Failed to compile the model: {e}")
    raise
file_name = f"drive/MyDrive/{model_name}.hef"
with open(file_name, "wb") as f:
    f.write(hef)