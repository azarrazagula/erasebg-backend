import os

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model_files"))

# Map the rembg model name to the actual ONNX filename
MODEL_REGISTRY = {
    "birefnet-general": "birefnet-general.onnx",
    "birefnet-portrait": "birefnet-portrait.onnx",
    "bria-rmbg": "bria-rmbg.onnx",
    "isnet-general-use": "isnet-general-use.onnx",
    "u2net": "u2net.onnx",
    "u2net_human_seg": "u2net_human_seg.onnx"
}

def get_model_path(model_name: str) -> str:
    """Returns the absolute path to the local ONNX model file."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} is not defined in the registry.")
    
    filename = MODEL_REGISTRY[model_name]
    return os.path.join(MODEL_DIR, filename)

def verify_model_exists(model_name: str) -> bool:
    """Checks if the model file is physically present in the model_files directory."""
    return os.path.exists(get_model_path(model_name))
