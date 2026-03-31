import os
import subprocess

# --- CONFIGURATION ---
INPUT_VIDEO = "merged_video.mp4" # The file we just created
CLIPS_TO_KEEP = "clips_to_keep.txt"
OUTPUT_DIR = "./processed"

def process_merged_video(video_id):
    """
    video_id: The ID you want to match in clips_to_keep (e.g., 'XMzg2MzQ1MTIyOA')
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"🎬 Processing Video ID: {video_id}")

    count = 0
    with open(CLIPS_TO_KEEP, 'r') as f:
        for line in f:
            segment = line.strip()
            if not segment: continue

            # Match segment name to your Video ID
            # Assuming clips_to_keep format: VideoID_Something_Index
            if video_id in segment:
                try:
                    # Extract the index (the last part after the last underscore)
                    parts = segment.split('_')
                    required_index = int(parts[-1])

                    # YOUR MAGIC FORMULA: start = (index - 1) * 2 + 1.5
                    start_time = (required_index - 1) * 2 + 1.5
                    if start_time < 0: start_time = 0

                    output_filename = f"{OUTPUT_DIR}/{segment}.mp4"

                    # FFmpeg command: Fast seeking (-ss before -i) + 3s duration
                    # We re-encode (libx264) to ensure frame-accurate cuts for your ML dataset
                    cmd = [
                        "ffmpeg", "-y",
                        "-ss", str(start_time),
                        "-i", INPUT_VIDEO,
                        "-t", "3",
                        "-c:v", "libx264",
                        "-c:a", "aac",
                        "-loglevel", "error",
                        output_filename
                    ]

                    subprocess.run(cmd)
                    count += 1
                    if count % 10 == 0:
                        print(f"✅ Processed {count} clips...")

                except Exception as e:
                    print(f"⚠️ Error on segment {segment}: {e}")

    print(f"✨ Done! Generated {count} clips for {video_id}.")

# --- RUN IT ---
# Replace this with the actual ID from your clips_to_keep.txt
target_id = input("Enter the Video ID to process: ")
process_merged_video(target_id)
