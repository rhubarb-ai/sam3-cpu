#!/usr/bin/env python3
"""
Validation script to check SAM3 CPU installation and structure.
"""

import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists."""
    if Path(filepath).exists():
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description} not found: {filepath}")
        return False

def check_dir_exists(dirpath, description):
    """Check if a directory exists."""
    if Path(dirpath).is_dir():
        print(f"✓ {description}: {dirpath}")
        return True
    else:
        print(f"✗ {description} not found: {dirpath}")
        return False

def count_files_in_dir(dirpath, pattern="*.py"):
    """Count files matching pattern in directory."""
    if Path(dirpath).is_dir():
        count = len(list(Path(dirpath).glob(pattern)))
        return count
    return 0

def main():
    print("=" * 70)
    print("SAM3 CPU - Installation Validation")
    print("=" * 70)
    print()
    
    all_checks_passed = True
    
    # Check core files
    print("Core Files:")
    all_checks_passed &= check_file_exists("sam3/__init__.py", "Package init")
    all_checks_passed &= check_file_exists("sam3/__globals.py", "Global config")
    all_checks_passed &= check_file_exists("sam3/entrypoint.py", "Main entrypoint (2038 lines)")
    all_checks_passed &= check_file_exists("sam3/wrapper.py", "Wrapper module")
    print()
    
    # Check directories
    print("Directories:")
    all_checks_passed &= check_dir_exists("examples", "Examples directory")
    all_checks_passed &= check_dir_exists("tests", "Tests directory")
    all_checks_passed &= check_dir_exists("assets/images", "Image assets")
    all_checks_passed &= check_dir_exists("assets/videos", "Video assets")
    print()
    
    # Check example files
    print("Example Scripts:")
    example_count = count_files_in_dir("examples", "example_*.py")
    if example_count >= 9:
        print(f"✓ Found {example_count} example scripts (expected 9+)")
    else:
        print(f"✗ Found only {example_count} example scripts (expected 9+)")
        all_checks_passed = False
    
    all_checks_passed &= check_file_exists("examples/run_all_examples.py", "Grouped runner")
    print()
    
    # Check test files
    print("Test Files:")
    all_checks_passed &= check_file_exists("tests/conftest.py", "Pytest config")
    all_checks_passed &= check_file_exists("tests/test_all_scenarios.py", "Scenario tests")
    print()
    
    # Check documentation and build files
    print("Documentation & Build:")
    all_checks_passed &= check_file_exists("README.md", "README")
    all_checks_passed &= check_file_exists("Makefile", "Makefile")
    all_checks_passed &= check_file_exists("setup.sh", "Setup script")
    all_checks_passed &= check_file_exists("pyproject.toml", "Project config")
    print()
    
    # Check imports
    print("Import Tests:")
    try:
        from sam3 import Sam3
        print("✓ Can import Sam3")
    except Exception as e:
        print(f"✗ Cannot import Sam3: {e}")
        all_checks_passed = False
    
    try:
        from sam3 import Sam3Entrypoint
        print("✓ Can import Sam3Entrypoint")
    except Exception as e:
        print(f"✗ Cannot import Sam3Entrypoint: {e}")
        all_checks_passed = False
    
    try:
        from sam3 import Sam3, Sam3Entrypoint
        if Sam3 is Sam3Entrypoint:
            print("✓ Sam3 is Sam3Entrypoint (alias works)")
        else:
            print("✗ Sam3 is not Sam3Entrypoint (alias broken)")
            all_checks_passed = False
    except Exception as e:
        print(f"✗ Alias check failed: {e}")
        all_checks_passed = False
    
    try:
        from sam3 import Sam3VideoPredictor
        print("✓ Can import Sam3VideoPredictor (backward compatibility)")
    except Exception as e:
        print(f"✗ Cannot import Sam3VideoPredictor: {e}")
        all_checks_passed = False
    
    print()
    
    # Check assets
    print("Assets:")
    image_count = count_files_in_dir("assets/images", "*.*")
    video_count = count_files_in_dir("assets/videos", "*.mp4")
    print(f"  Images: {image_count} files")
    print(f"  Videos: {video_count} files")
    print()
    
    # Final result
    print("=" * 70)
    if all_checks_passed:
        print("✓ All validation checks passed!")
        print()
        print("Next steps:")
        print("  1. Run examples: make run-all")
        print("  2. Run tests: make test")
        print("  3. See README.md for full documentation")
        return 0
    else:
        print("✗ Some validation checks failed!")
        print("Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
