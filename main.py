from sam3.entry import Sam3Entry


if __name__ == "__main__":
    # video_file = "assets/videos/Roger-Federer-vs-Rafael-Nadal-Wimbledon-2008_480p.mp4"
    video_file = "assets/videos/sample.mp4"
    prompt = ["player", "ball", "tennis court", "net"]

    entry = Sam3Entry()
    result = entry.process_video_with_prompts(video_file, prompt)
    
