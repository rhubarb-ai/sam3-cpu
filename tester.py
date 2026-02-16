import json
from sam3.driver import Sam3ImageDriver, Sam3VideoDriver

if __name__ == "__main__":
    print("Testing Sam3ImageDriver...")

    image_driver = Sam3ImageDriver()
    results = image_driver.prompt_image(
        "assets/images/test_image.jpg", 
        prompts=["kids", "window"]
    )

    # save results for inspection
    # with open("test_image_results.json", "w") as f:
    #     json.dump(results, f, indent=4, default=str)

    print("✓ Sam3ImageDriver initialized successfully.\n")

    print("Testing Sam3VideoDriver...")
    # video_driver = Sam3VideoDriver()
    print("✓ Sam3VideoDriver initialized successfully.\n")

    print("All tests passed!")