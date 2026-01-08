DEBUG = False

LABELS = ['up', 'down', 'left', 'right', 'other', 'silence']
ACTION_LABELS = ["up", "down", "left", "right"]

SAMPLE_RATE = 16000

INFERENCE_INTERVAL = 0.01  # Run inference every 10ms
CONFIDENCE_THRESHOLD = 0.95

MODEL_PATH = "model.onnx"
