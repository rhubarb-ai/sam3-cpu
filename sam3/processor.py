
from pathlib import Path
from sam3.__globals import logger, DEVICE, DEFAULT_OUTPUT_DIR, TEMP_DIR


class ImageProcessor:
    def __init__(self):
        pass

    


class ChunkProcessor:
    def __init__(self):
        self.chunk_meta_path = None
        self.chunk_mask_dir = None



class VideoProcessor:
    def __init__(
        self, 
        video_path: str | Path, 
        output_dir: str | Path = Path(DEFAULT_OUTPUT_DIR), 
        temp_dir: str | Path = Path(TEMP_DIR)
    ):
        self.video_meta_path = None
        self.__initialize_video(video_path, output_dir, temp_dir)

    def __initialize_video(
        self, 
        video_path: str | Path, 
        output_dir: str | Path, 
        temp_dir: str | Path
    ):
        video_name = Path(video_path).stem
        self._create_temp_directories(video_name, temp_dir)
        self._create_output_directories(video_name, output_dir)

    def _create_output_directories(
        self, 
        video_name: str, 
        base_dir: str | Path = Path(DEFAULT_OUTPUT_DIR)
    ):
        if not base_dir:
            base_dir = Path(DEFAULT_OUTPUT_DIR)
        if type(base_dir) == str:
            base_dir = Path(base_dir)

        self.video_dir = base_dir / video_name
        self.video_dir.mkdir(parents=True, exist_ok=True)

        self.video_meta_dir = self.video_dir / "metadata"
        self.video_meta_dir.mkdir(parents=True, exist_ok=True)

        self.video_mask_dir = self.video_dir / "masks"
        self.video_mask_dir.mkdir(parents=True, exist_ok=True)

        self.video_results_dir = self.video_dir / "results"
        self.video_results_dir.mkdir(parents=True, exist_ok=True)

    def _create_temp_directories(self, video_name: str, temp_dir: str | Path = Path(TEMP_DIR)):
        if not temp_dir:
            temp_dir = Path(TEMP_DIR)
        if type(temp_dir) == str:
            temp_dir = Path(temp_dir)

        self.video_temp_dir = temp_dir / video_name
        self.video_temp_dir.mkdir(parents=True, exist_ok=True)

        # Create chunks subdirectory
        self.chunks_temp_dir = self.video_temp_dir / "chunks"
        self.chunks_temp_dir.mkdir(parents=True, exist_ok=True)

        # Create metadata subdirectory
        self.meta_temp_dir = self.video_temp_dir / "metadata"
        self.meta_temp_dir.mkdir(parents=True, exist_ok=True)

    def process_bbox(self):
        pass

    def process_clicks(self):
        pass

    def process_prompts(self, prompts: str | List[str], device: str = DEVICE):
        pass


