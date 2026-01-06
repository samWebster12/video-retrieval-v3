from scenedetect import open_video, SceneManager
from scenedetect.detectors import AdaptiveDetector

def detect_shots_pyscenedetect(video_path, threshold=3.0, min_scene_len=15):
    """
    Detects shots using PySceneDetect's AdaptiveDetector.
    
    Args:
        video_path (str): Path to the video file.
        threshold (float): 'adaptive_threshold'. How much the frame must exceed the 
                           rolling average to trigger a cut. 
                           3.0 is standard. Lower (2.0) is more sensitive.
        min_scene_len (int): Minimum length of a scene in frames. 
                             Prevents detecting cuts too close together (debouncing).
    """
    # 1. Open the video
    # scenedetect handles OpenCV/FFmpeg backend automatically
    try:
        video = open_video(video_path)
    except Exception as e:
        print(f"Could not open video: {e}")
        return []

    # 2. Create a SceneManager and add the AdaptiveDetector
    scene_manager = SceneManager()
    
    # window_width=25 is standard (rolling window size)
    scene_manager.add_detector(AdaptiveDetector(
        adaptive_threshold=threshold, 
        min_scene_len=min_scene_len,
        window_width=100 
    ))

    # 3. Process the video
    # show_progress=True prints a progress bar to terminal
    print(f"Processing {video_path}...")
    scene_manager.detect_scenes(video, show_progress=False)

    # 4. Get list of scenes
    # Returns a list of tuples: (start_time, end_time)
    scene_list = scene_manager.get_scene_list()

    # 5. Convert to your desired list format: [(Timestamp, Frame, FPS), ...]
    # Note: PySceneDetect doesn't easily expose the raw "score" for every cut 
    # in the simple API, so we return FPS as the 3rd value (or you could put None).
    shot_boundaries = []

    for i, scene in enumerate(scene_list):
        # The 'start' of a scene is the Cut point.
        # We skip the very first scene start (00:00:00) as it's not a "cut".
        if i == 0:
            continue
            
        timecode = scene[0] # This is a FrameTimecode object
        
        timestamp = timecode.get_seconds()
        frame_num = timecode.get_frames()
        
        shot_boundaries.append((timestamp, frame_num))

    return shot_boundaries

if __name__ == "__main__":
    video_file = "assets/worldcup/40d78e0a05954fefa8ae546d7214923f.mp4"
    
    shots = detect_shots_pyscenedetect(video_file, threshold=2.5)
    
    print(f"\nDetected {len(shots)} shots:")
    print(f"{'Time':<10} | {'Frame':<10} | {'FPS':<10}")
    print("-" * 35)
    for s in shots:
        print(f"{s[0]:<10.4f} | {s[1]:<10} | {s[2]:<10.4f}")