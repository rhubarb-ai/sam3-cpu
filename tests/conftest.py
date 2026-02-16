"""
Pytest configuration and shared fixtures for SAM3 tests.
"""

import pytest
from pathlib import Path
import tempfile
import shutil


@pytest.fixture(scope="session")
def assets_dir():
    """Path to assets directory."""
    return Path(__file__).parent.parent / "assets"


@pytest.fixture(scope="session")
def test_image_truck(assets_dir):
    """Path to truck test image."""
    path = assets_dir / "images" / "truck.jpg"
    if not path.exists():
        pytest.skip(f"Test image not found: {path}")
    return str(path)


@pytest.fixture(scope="session")
def test_image_cafe(assets_dir):
    """Path to cafe test image."""
    path = assets_dir / "images" / "cafe.png"
    if not path.exists():
        pytest.skip(f"Test image not found: {path}")
    return str(path)


@pytest.fixture(scope="session")
def test_image_groceries(assets_dir):
    """Path to groceries test image."""
    path = assets_dir / "images" / "groceries.jpg"
    if not path.exists():
        pytest.skip(f"Test image not found: {path}")
    return str(path)


@pytest.fixture(scope="session")
def test_image_test(assets_dir):
    """Path to generic test image."""
    path = assets_dir / "images" / "test_image.jpg"
    if not path.exists():
        pytest.skip(f"Test image not found: {path}")
    return str(path)


@pytest.fixture(scope="session", params=["480p", "720p", "1080p"])
def test_video_resolution(request):
    """Video resolution parameter."""
    return request.param


@pytest.fixture(scope="session")
def test_video_tennis_480p(assets_dir):
    """Path to tennis 480p test video."""
    path = assets_dir / "videos" / "Roger-Federer-vs-Rafael-Nadal-Wimbledon-2008_480p.mp4"
    if not path.exists():
        pytest.skip(f"Test video not found: {path}")
    return str(path)


@pytest.fixture(scope="session")
def test_video_tennis_720p(assets_dir):
    """Path to tennis 720p test video."""
    path = assets_dir / "videos" / "Roger-Federer-vs-Rafael-Nadal-Wimbledon-2008_720p.mp4"
    if not path.exists():
        pytest.skip(f"Test video not found: {path}")
    return str(path)


@pytest.fixture(scope="session")
def test_video_tennis_1080p(assets_dir):
    """Path to tennis 1080p test video."""
    path = assets_dir / "videos" / "Roger-Federer-vs-Rafael-Nadal-Wimbledon-2008_1080p.mp4"
    if not path.exists():
        pytest.skip(f"Test video not found: {path}")
    return str(path)


@pytest.fixture(scope="session")
def test_video_bedroom(assets_dir):
    """Path to bedroom test video."""
    path = assets_dir / "videos" / "bedroom.mp4"
    if not path.exists():
        pytest.skip(f"Test video not found: {path}")
    return str(path)


@pytest.fixture(scope="function")
def temp_output_dir():
    """Create a temporary output directory for each test."""
    temp_dir = tempfile.mkdtemp(prefix="sam3_test_")
    yield temp_dir
    # Cleanup after test
    if Path(temp_dir).exists():
        shutil.rmtree(temp_dir)


@pytest.fixture(scope="session")
def sam3_instance():
    """Create a SAM3 instance for testing."""
    from sam3 import Sam3
    return Sam3(verbose=True)


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "image: marks tests for image processing")
    config.addinivalue_line("markers", "video: marks tests for video processing")
    config.addinivalue_line("markers", "scenario_a: single image with text prompts")
    config.addinivalue_line("markers", "scenario_b: single image with bounding boxes")
    config.addinivalue_line("markers", "scenario_c: batch images with text prompts")
    config.addinivalue_line("markers", "scenario_d: batch images with bounding boxes")
    config.addinivalue_line("markers", "scenario_e: video with text prompts")
    config.addinivalue_line("markers", "scenario_f: video with point prompts")
    config.addinivalue_line("markers", "scenario_g: refine video object")
    config.addinivalue_line("markers", "scenario_h: remove video objects")
    config.addinivalue_line("markers", "scenario_i: video with segments")
