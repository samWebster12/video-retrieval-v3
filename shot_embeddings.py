import torch
import numpy as np
from torch.nn.functional import normalize
from tqdm import tqdm
from typing import List

def process_shots_to_embeddings(
    shot_tensors: List[torch.Tensor],
    model,
    device,
    batch_size: int = 16
) -> np.ndarray:
    """
    Takes a list of shot tensors (T, C, H, W), runs VideoMAEv2 inference, 
    and returns normalized embeddings. Matches reference logic exactly.
    """
    
    if not shot_tensors:
        return np.empty((0, 0)) 

    if device != "cpu" and torch.cuda.device_count() > 1:
        if not isinstance(model, torch.nn.DataParallel):
            device_ids = list(range(torch.cuda.device_count()))
            model = torch.nn.DataParallel(model, device_ids=device_ids)
    
    model.to(device)
    model.eval()

    embeddings_list = []
    
    for i in tqdm(range(0, len(shot_tensors), batch_size), unit="batch"):
        batch_tensors = shot_tensors[i : i + batch_size]
        stacked_batch = torch.stack([t.permute(1, 0, 2, 3) for t in batch_tensors])
        inputs = stacked_batch.to(device, dtype=torch.float32, non_blocking=True)

        with torch.no_grad():
            vec = model(pixel_values=inputs)

        v = normalize(vec, dim=-1).cpu().numpy()
            
        embeddings_list.append(v)

    if embeddings_list:
        final_embeddings = np.concatenate(embeddings_list, axis=0)
    else:
        final_embeddings = np.array([])

    return final_embeddings

# ==========================================
# Full Pipeline Example
# ==========================================
if __name__ == "__main__":
    mock_shots_detected = [
        (4.5, 135, 30.0), 
        (8.2, 246, 30.0), 
        (12.0, 360, 30.0)
    ] 
    video_path = "assets/bm/904c8ebf782357ae78ebd205fe3428ad76b975a5.mp4"

    try:
        from shot_frame_extraction import extract_frames_per_shot
        shot_tensors = extract_frames_per_shot(video_path, mock_shots_detected)
    except NameError:
        print("extract_frames_per_shot not found, generating mock tensors...")
        shot_tensors = [torch.randn(16, 3, 224, 224) for _ in range(3)]

    print("Loading Model...")

    from load_videomae_model import load_model  
    model = load_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    embeddings = process_shots_to_embeddings(
        shot_tensors, 
        model, 
        device=device,
        batch_size=8
    )

    print(f"\nPipeline Complete.")
    print(f"Input Shots: {len(shot_tensors)}")
    print(f"Output Embeddings Shape: {embeddings.shape}") 
    # Expected: (3, 768)
    
    print("\nFirst embedding vector (first 5 vals):")
    print(embeddings[0][:5])