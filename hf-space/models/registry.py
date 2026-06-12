import os

# model_files/ directory absolute path — __file__ = registry.py இருக்கற இடம்
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model_files"))

# rembg model name → actual ONNX filename mapping
# rembg library இந்த names-ஐ use பண்ணும், நாம் local file-ஐ point பண்றோம்
MODEL_REGISTRY = {
    "birefnet-general": "birefnet-general.onnx",        # General objects, graphics
    "birefnet-portrait": "birefnet-portrait.onnx",      # Humans, portraits
    "bria-rmbg": "bria-rmbg.onnx",                      # Alternative model (optional)
    "isnet-general-use": "isnet-general-use.onnx",      # ISNet model (optional)
    "u2net": "u2net.onnx",                               # Original rembg model (optional)
    "u2net_human_seg": "u2net_human_seg.onnx"           # Human segmentation (optional)
}

def get_model_path(model_name: str) -> str:
    """Returns the absolute path to the local ONNX model file."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} is not defined in the registry.")
    
    filename = MODEL_REGISTRY[model_name]           # ONNX filename எடு
    return os.path.join(MODEL_DIR, filename)        # Full path return பண்ணு

def verify_model_exists(model_name: str) -> bool:
    """Checks if the model file is physically present in the model_files directory."""
    return os.path.exists(get_model_path(model_name))  # File disk-ல இருக்கான்னு check
