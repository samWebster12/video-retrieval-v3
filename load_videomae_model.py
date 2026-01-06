import torch
from transformers import AutoModel, AutoConfig

MODEL_CKPT = "OpenGVLab/VideoMAEv2-Base"
WINDOW_FRAMES = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
def load_model():
    print(f"Loading OpenGVLab model: {MODEL_CKPT}...")

    config = AutoConfig.from_pretrained(MODEL_CKPT, trust_remote_code=True)
    config.model_config["num_frames"] = WINDOW_FRAMES 

    # Load Model
    if DEVICE != "cpu":
        model = AutoModel.from_pretrained(
            MODEL_CKPT, 
            config=config, 
            trust_remote_code=True, 
            ignore_mismatched_sizes=True
        ).eval().cuda()
    else:
        model = AutoModel.from_pretrained(
            MODEL_CKPT, 
            config=config, 
            trust_remote_code=True, 
            ignore_mismatched_sizes=True
        ).eval().to(DEVICE)
    
    return model