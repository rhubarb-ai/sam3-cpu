# from sam3.entry import Sam3Entry


# if __name__ == "__main__":
#     # video_file = "assets/videos/Roger-Federer-vs-Rafael-Nadal-Wimbledon-2008_480p.mp4"
#     video_file = "assets/videos/sample.mp4"
#     prompt = ["player", "ball", "tennis court", "net"]

#     entry = Sam3Entry()
#     result = entry.process_video_with_prompts(video_file, prompt)
    
from sam3 import Sam3API

# Initialize
api = Sam3API()

# Process video
result = api.process_video_with_prompts(
    video_path="assets/videos/sample.mp4",
    prompts=["player", "ball", "tennis court"],
    keep_temp_files=True  # or True to preserve
)

# Process image
result = api.process_image_with_prompts(
    image_path="assets/images/test_image.jpg",
    prompts=["kids", "window", "hairband"]
)

# Always cleanup
api.cleanup()