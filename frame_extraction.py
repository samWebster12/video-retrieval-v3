import subprocess
import torch
import numpy as np
import warnings
import json

# Constants from your reference code
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

def get_video_duration(video_path):
    """Get exact video duration using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    try:
        return float(subprocess.check_output(cmd).decode().strip())
    except Exception:
        return 0.0

def extract_frames_per_shot(
    video_path: str, 
    shot_boundaries: list, 
    num_frames_per_shot: int = 16
) -> list:
    """
    Extracts 'num_frames_per_shot' equally spaced frames from each detected shot.
    
    Args:
        video_path: Path to video.
        shot_boundaries: Output from detect_shots (list of tuples or list of timestamps).
                         We expect a list of timestamps indicating CUTS.
        num_frames_per_shot: Number of frames to extract per shot (default 16).
        
    Returns:
        List[torch.Tensor]: A list where each element is a Tensor of shape (16, 3, 224, 224).
    """
    
    # 1. Prepare Shot Intervals (Start, End)
    # The boundaries are usually cut points. 
    # Shot 1: 0.0 -> Cut 1
    # Shot 2: Cut 1 -> Cut 2
    # Last Shot: Cut N -> Video End
    
    duration = get_video_duration(video_path)
    if duration == 0:
        raise ValueError("Could not determine video duration.")

    # Extract just the timestamps if the input is list of tuples
    # (Handling both [time, time...] and [(time, frame, score)...])
    cuts = []
    if shot_boundaries and isinstance(shot_boundaries[0], (tuple, list)):
        cuts = [x[0] for x in shot_boundaries]
    else:
        cuts = shot_boundaries

    # Sort and filter unique
    cuts = sorted(list(set(cuts)))
    
    intervals = []
    current_time = 0.0
    
    for cut_time in cuts:
        # Avoid creating shots with 0 duration or negative duration
        if cut_time > current_time:
            intervals.append((current_time, cut_time))
        current_time = cut_time
        
    # Append the final shot (from last cut to end of video)
    if current_time < duration:
        intervals.append((current_time, duration))

    results = []

    print(f"Processing {len(intervals)} shots for video: {video_path}")

    # 2. Extract frames for each interval
    for idx, (start, end) in enumerate(intervals):
        shot_duration = end - start
        
        # Safety check for extremely short shots
        if shot_duration <= 0.05: 
            continue

        # Calculate exact FPS needed to get 'num_frames' within 'shot_duration'
        # e.g., if shot is 2 seconds and we need 16 frames, fps = 8.
        target_fps = num_frames_per_shot / shot_duration

        # Construct Filter Chain (Exact match to your reference)
        # 1. fps filter sets the rate
        # 2. scale/crop handles the 224x224 center crop
        vf = (
            f"fps={target_fps},"
            "scale='if(gt(iw,ih),-1,224)':'if(gt(iw,ih),224,-1)',"
            "crop=224:224:(iw-224)/2:(ih-224)/2"
        )

        # Efficient Seeking:
        # -ss before -i is "Input Seeking" (very fast, jumps to keyframe)
        # -t limits the duration read
        cmd = [
            "ffmpeg", "-v", "error",
            "-ss", f"{start:.4f}",
            "-t", f"{shot_duration:.4f}",
            "-i", video_path,
            "-vf", vf,
            "-sws_flags", "bilinear",
            "-pix_fmt", "rgb24",
            "-f", "rawvideo", 
            "pipe:1"
        ]

        try:
            # Run FFmpeg
            raw = subprocess.check_output(cmd)
            
            # 3. Process Bytes to Tensor
            frame_bytes = 224 * 224 * 3
            if len(raw) == 0:
                print(f"Warning: Shot {idx} returned empty.")
                continue

            n_total = len(raw) // frame_bytes
            
            # Reshape raw bytes
            arr = np.frombuffer(raw, np.uint8).reshape(n_total, 224, 224, 3)

            # 4. Handle Frame Count Mismatch
            # FFmpeg's 'fps' filter is approximate. We might get 15 or 17 frames.
            # We enforce exactly 'num_frames_per_shot'.
            
            if n_total > num_frames_per_shot:
                # Truncate extra frames
                arr = arr[:num_frames_per_shot]
            elif n_total < num_frames_per_shot:
                # Pad with the last frame if we are short
                diff = num_frames_per_shot - n_total
                if n_total > 0:
                    last_frame = arr[-1:]
                    padding = np.repeat(last_frame, diff, axis=0)
                    arr = np.concatenate((arr, padding), axis=0)
                else:
                    # If shot was valid but yielded 0 frames (rare), skip
                    continue

            # 5. Convert to Torch & Normalize (Exact reference logic)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                # Normalize 0-255 -> 0.0-1.0
                t = torch.from_numpy(arr.copy()).float().div_(255.0)

            # Permute: (N, H, W, C) -> (N, C, H, W)
            t = t.permute(0, 3, 1, 2)

            # Manual Normalization (ImageNet)
            for c, (m, s) in enumerate(zip(_IMAGENET_MEAN, _IMAGENET_STD)):
                t[:, c].sub_(m).div_(s)

            results.append(t)

        except subprocess.CalledProcessError as e:
            print(f"Error processing shot {start}-{end}: {e}")
            continue

    return results

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # 1. Assume we ran the previous pyscenedetect function
    # Mocking output for demonstration:
    # (timestamp, frame, fps)
    mock_shots = [
        (4.5, 135, 30.0),  # Cut at 4.5s
        (8.2, 246, 30.0),  # Cut at 8.2s
        (12.0, 360, 30.0)  # Cut at 12.0s
    ]
    
    video_file = "assets/bm/904c8ebf782357ae78ebd205fe3428ad76b975a5.mp4"
    
    try:
        # Returns list of Tensors
        shot_tensors = extract_frames_per_shot(video_file, mock_shots, num_frames_per_shot=16)
        
        print(f"\nExtracted {len(shot_tensors)} shot tensors.")
        
        if len(shot_tensors) > 0:
            print(f"Shape of first shot tensor: {shot_tensors[0].shape}") 
            # Expected: torch.Size([16, 3, 224, 224])
            
            print(f"Data Type: {shot_tensors[0].dtype}")
            # Expected: torch.float32
            
    except Exception as e:
        print(f"Pipeline failed: {e}")