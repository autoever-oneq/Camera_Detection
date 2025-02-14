import json
import os
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive/', force_remount=True)

# Updated NMS layer configuration dictionary
nms_layer_config = {
	"nms_scores_th": 0.2,
	"nms_iou_th": 0.7,
	"image_dims": [
		640,
		640
	],
	"max_proposals_per_class": 2,
	"classes": 2,
	"regression_length": 16,
	"background_removal": False,
	"bbox_decoders": [
		{
			"name": "yolov8n/bbox_decoder41",
			"stride": 8,
			"reg_layer": "yolov8n/conv41",
			"cls_layer": "yolov8n/conv42"
		},
		{
			"name": "yolov8n/bbox_decoder52",
			"stride": 16,
			"reg_layer": "yolov8n/conv52",
			"cls_layer": "yolov8n/conv53"
		},
		{
			"name": "yolov8n/bbox_decoder62",
			"stride": 32,
			"reg_layer": "yolov8n/conv62",
			"cls_layer": "yolov8n/conv63"
		}
	]
}

# Path to save the updated JSON configuration
output_dir = "/save/path/"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
output_path = os.path.join(output_dir, "nms_layer_config.json")

# Save the updated configuration as a JSON file
with open(output_path, "w") as json_file:
    json.dump(nms_layer_config, json_file, indent=4)

print(f"NMS layer configuration saved to {output_path}")