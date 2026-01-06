import os
import subprocess
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, Optional, Any

# ==========================================
# 1. The Two Major Functions
# ==========================================

def get_isc_features(
    video_path: str, 
    extractor: Optional["ISCFeatureExtractor"] = None,
    fps: float = 1.0, 
    device: str = "cuda"
) -> np.ndarray:
    """
    Extracts ISC features from a video.
    
    Args:
        video_path: Path to mp4/avi file.
        extractor: Instance of ISCFeatureExtractor. If None, one is created (slower).
        fps: Frames per second to extract.
        device: 'cuda' or 'cpu' (only used if extractor is None).
        
    Returns:
        np.ndarray: Shape (N, 256) float32 features.
    """
    # Load model on the fly if not provided (convenient but inefficient for loops)
    if extractor is None:
        extractor = ISCFeatureExtractor(device=device)
        
    return extractor.process_video(video_path, fps=fps)


def save_isc_features(features: np.ndarray, output_path: str) -> None:
    """
    Saves ISC features to a .npy file, creating directories if needed.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure standard extension
    if out_p.suffix != '.npy':
        out_p = out_p.with_suffix('.npy')
        
    np.save(str(out_p), features)
    print(f"[Saved] {features.shape} -> {out_p}")


# ==========================================
# 2. The Engine (Class & Helpers)
# ==========================================

class ISCFeatureExtractor:
    """
    Handles model loading and inference to avoid reloading weights 
    for every video.
    """
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocessor = self._load_model()

    def _load_model(self) -> Tuple[Any, Any]:
        try:
            from isc_feature_extractor import create_model
        except ImportError:
            raise ImportError(
                "ISC feature extractor not installed.\n"
                "pip install git+https://github.com/lyakaap/ISC21-Descriptor-Track-1st"
            )
        
        print(f"[ISC] Loading model on {self.device}...")
        model, preprocessor = create_model(weight_name="isc_ft_v107", device=self.device)
        model.eval()
        return model, preprocessor

    @torch.no_grad()
    def process_video(self, video_path: str, fps: float = 1.0, batch_size: int = 32) -> np.ndarray:
        """Main pipeline: Extract Frames -> Preprocess -> Inference"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        # 1. Extract raw frames using FFmpeg
        frames_rgb = _ffmpeg_extract_frames(video_path, fps)
        
        if len(frames_rgb) == 0:
            print(f"[Warn] No frames extracted from {video_path}")
            return np.empty((0, 256), dtype=np.float32)

        # 2. Batch Inference
        feats_list = []
        for i in tqdm(range(0, len(frames_rgb), batch_size), desc="Extracting ISC", leave=False):
            batch_arr = frames_rgb[i : i + batch_size]
            
            # Preprocess (Numpy -> PIL -> Tensor)
            batch_tensors = []
            for frame in batch_arr:
                pil_img = Image.fromarray(frame)
                tensor_img = self.preprocessor(pil_img)
                batch_tensors.append(tensor_img)
            
            # Stack and Infer
            input_batch = torch.stack(batch_tensors).to(self.device)
            embeddings = self.model(input_batch) # Model outputs L2 normalized features
            feats_list.append(embeddings.cpu().numpy())

        # 3. Concatenate
        if not feats_list:
            return np.empty((0, 256), dtype=np.float32)
            
        return np.concatenate(feats_list, axis=0).astype(np.float32)


# ==========================================
# 3. FFmpeg Helpers (Private)
# ==========================================

def _get_video_dims(video_path: str) -> Tuple[int, int]:
    """Returns (width, height) using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        video_path
    ]
    try:
        out = subprocess.check_output(cmd).decode().strip()
        w, h = map(int, out.split(","))
        return w, h
    except Exception as e:
        raise RuntimeError(f"FFprobe failed for {video_path}: {e}")

def _calc_resize_dims(w: int, h: int, max_dim: int = 512) -> Tuple[int, int]:
    """Calculates new dims maintaining aspect ratio, max 512px, even numbers."""
    scale = min(max_dim / w, max_dim / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # FFmpeg requires even dimensions for some codecs/formats
    new_w -= new_w % 2
    new_h -= new_h % 2
    return max(new_w, 2), max(new_h, 2)

def _ffmpeg_extract_frames(video_path: str, fps: float) -> np.ndarray:
    """Returns Numpy array (N, H, W, 3) of RGB frames."""
    w, h = _get_video_dims(video_path)
    out_w, out_h = _calc_resize_dims(w, h)

    cmd = [
        "ffmpeg", "-v", "error",
        "-i", video_path,
        "-vf", f"fps={fps},scale={out_w}:{out_h}",
        "-pix_fmt", "rgb24",
        "-f", "rawvideo", 
        "pipe:1"
    ]
    
    try:
        # Increase buffer size for large videos
        raw_output = subprocess.check_output(cmd, bufsize=10**7)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg failed: {e}")

    # Calculate number of frames based on byte size
    frame_size = out_w * out_h * 3
    if len(raw_output) == 0:
        return np.array([])
        
    num_frames = len(raw_output) // frame_size
    
    # Truncate partial bytes if any
    valid_bytes = num_frames * frame_size
    raw_output = raw_output[:valid_bytes]
    
    return np.frombuffer(raw_output, dtype=np.uint8).reshape(num_frames, out_h, out_w, 3)


# ==========================================
# 4. Usage Example (CLI)
# ==========================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean ISC Feature Extractor")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--out", required=True, help="Output .npy path")
    parser.add_argument("--fps", type=float, default=1.0)
    args = parser.parse_args()

    # 1. Initialize logic once
    extractor = ISCFeatureExtractor()

    # 2. Get Features
    print(f"Processing {args.video}...")
    feats = get_isc_features(args.video, extractor=extractor, fps=args.fps)

    # 3. Save Features
    save_isc_features(feats, args.out)