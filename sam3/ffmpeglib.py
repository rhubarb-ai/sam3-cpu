import json
from sam3.utils import run_cmd

class FFMpegLib:
    def __init__(self):
        pass

    @staticmethod
    def get_video_info(video_file):
        """Get video information using ffprobe."""
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,duration,nb_frames",
            "-of", "json",
            video_file
        ]
        output = run_cmd(cmd)
        info = json.loads(output)
        info = info['streams'][0] if 'streams' in info and len(info['streams']) > 0 else None
        
        if info:
            info['duration'] = float(info['duration'])
            info['nb_frames'] = int(info['nb_frames'])

            # Parse FPS fraction safely
            info['fps'] = info['nb_frames']/info['duration'] if info['duration'] > 0 else 0
        
        return info
    
    @staticmethod
    def convert_to_constant_fps(input_video, output_video, target_fps=25):
        """Convert video to constant FPS using ffmpeg."""
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(input_video),
            "-r", str(target_fps),
            "-fps_mode", "cfr",
            "-an",  # REMOVE AUDIO STREAM
            str(output_video)
        ]
        output = run_cmd(cmd)
        return output
    
    @staticmethod
    def create_video_chunk(input_video, output_video, start_frame, end_frame):
        """Create a video chunk using ffmpeg."""
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(input_video),
            "-map", "0:v:0",
            "-vf", f"select='between(n\\,{start_frame}\\,{end_frame})',setpts=PTS-STARTPTS",
            "-c:v", "libx264",
            "-an",  # REMOVE AUDIO STREAM
            "-preset", "veryfast",
            str(output_video)
        ]
        output = run_cmd(cmd)
        return output
    
ffmpeg_lib = FFMpegLib()