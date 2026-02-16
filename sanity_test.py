#!/usr/bin/env python3
"""
Quick sanity test - create a Sam3 instance and check it initializes correctly.
This doesn't run actual inference, just validates the API structure.
"""

from sam3 import Sam3

def test_initialization():
    """Test that Sam3 can be initialized."""
    print("Testing Sam3 initialization...")
    
    try:
        sam3 = Sam3(verbose=False)
        print(f"✓ Sam3 instance created")
        print(f"  Device type: {sam3.memory_info.device_type}")
        print(f"  Total memory: {sam3.memory_info.total_gb:.2f} GB")
        print(f"  Available memory: {sam3.memory_info.available_gb:.2f} GB")
        print(f"  RAM usage percent: {sam3.ram_usage_percent}")
        print(f"  Min video frames: {sam3.min_video_frames}")
        print(f"  Num workers: {sam3.num_workers}")
        return True
    except Exception as e:
        print(f"✗ Failed to create Sam3 instance: {e}")
        return False

def test_api_methods():
    """Test that all API methods exist."""
    print("\nTesting API methods exist...")
    
    sam3 = Sam3(verbose=False)
    
    methods = [
        'process_image',
        'process_video_with_prompts',
        'process_video_with_points',
        'refine_video_object',
        'remove_video_objects',
        'process_video_with_segments'
    ]
    
    all_exist = True
    for method in methods:
        if hasattr(sam3, method):
            print(f"✓ {method}()")
        else:
            print(f"✗ {method}() not found")
            all_exist = False
    
    return all_exist

def test_import_aliases():
    """Test that import aliases work."""
    print("\nTesting import aliases...")
    
    try:
        from sam3 import Sam3, Sam3Entrypoint
        assert Sam3 is Sam3Entrypoint, "Sam3 should be same as Sam3Entrypoint"
        print("✓ Sam3 is Sam3Entrypoint")
        
        from sam3 import Sam3VideoPredictor
        print("✓ Sam3VideoPredictor available (backward compatibility)")
        
        from sam3 import Sam3Wrapper
        print("✓ Sam3Wrapper available")
        
        return True
    except Exception as e:
        print(f"✗ Import alias test failed: {e}")
        return False

def main():
    print("=" * 70)
    print("SAM3 CPU - Quick Sanity Test")
    print("=" * 70)
    print()
    
    results = []
    
    results.append(test_initialization())
    results.append(test_api_methods())
    results.append(test_import_aliases())
    
    print()
    print("=" * 70)
    
    if all(results):
        print("✓ All sanity tests passed!")
        print()
        print("SAM3 CPU is ready to use.")
        print("Run 'make run-all' to test all scenarios with actual inference.")
        return 0
    else:
        print("✗ Some sanity tests failed!")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
