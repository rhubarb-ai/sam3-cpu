- sam3/
    - Create sam3/utils folder and move utilities related files and functions here. Make sure to correct cross referencings.
    - Clean up unwanted files and functions and move them into sam3/archive folder for now (don't delete them permanently)
- test files at the root level
    - read them thoroughly and we have to possibly convert them into a few master files after combining the logics of similar ones.

- we create two main files at the root level:
    - image_prompter.py
    - video_prompter.py

- image_prompter can take a single or multiple images with: 
    -- [optional] single or multiple text prompts, 
    -- [optional] single or multiple click points as (x,y) co-ord in the image, and 
    -- [optional] bounding box, that can be 4 coords of rect or diagonal coords as 2 coords. 
    -- Also [optional] output folder can be provided, 
    -- [optional] alpha value on the overlay when we overlay the masks on the image.
    -- [optional] device - user choice of their device - they can still run on cpu even though if their device has GPU support

    -- If multiple images, loop through each image:
        Prior to Loading Images - Validations:
        -- Validate for: At least one of these optional (prompts/click points/bbox) inputs must be provided otherwise exit the program by giving the error message. 
        -- Memory Validation: Run the memory tests and check if there is sufficient memory to run this image or not. Show the use memory related output in a nicer (perhaps tabular) way.
        -- If the image can't be processed due to memory constraint show the error with how much memory is missing in red.
        -- All these memory related data will go into image metadata file - so cache it properly.

        After Memory validation passes:
        -- Multiple text prompts can go into the loop while I think others can be processed simultaneously. 
        -- If the text prompts are given then we save the data into [output_folder]/[images]/[image_name]/[prompt_text]/ folder. Here we save all data including masks, overlay, and any json results as the image metadata and processed features from the image etc.
        -- If the text prompts were not not given then we save the data into [output_folder]/[images]/[image_name]/ folder alike how we are saving above.

- video_prompter will take a single video:
    -- [optional] single or multiple text prompts,
    -- [optional] single or multiple click points as (x,y) co-ord in the video, and
    -- [optional] single or multiple masks location
    -- Also [optional] output folder can be provided, 
    -- [optional] alpha value on the overlay when we overlay the masks on the video. 
    -- [optional] device - user choice of their device - they can still run on cpu even though if their device has GPU support

    Prior to loading video:
    -- Validate for: At least one of these optional (prompts/click points/masks) inputs must be provided otherwise exit the program by giving the error message.
    -- Validation: When masks location is given - the size of masks must match with the video dim
    -- Memory Validation: Run the memory tests and check if there is sufficient memory to run this video with the minimum number of frames (DEFAULT_MIN_VIDEO_FRAMES) or not. Show the use memory related output in a nicer (perhaps tabular) way.
    -- If the video can't be processed due to memory constraint show the error with how much memory is missing in red.
    -- All these memory related data will go into video metadata file - so cache it properly.

    After Memory validation passes:
    -- Video chunking mechanism: Run video chunking strategy. We'll chunk the video with overlapping frames (DEFAULT_MIN_CHUNK_OVERLAP) one by one and save the chunks inside temp folder (TEMP_DIR) with the video_name and so on. We already did in the past - so you can take things from there.
    -- This script is the best example how chunked videos can be processed: notebook/video_chunked_predictor.example.ipynb utilise this at the best.
    -- Multiple text prompts can go into the loop while I think others can be processed simultaneously. 
    -- All intermediate data including masks and metadata should be first stored inside TEMP_DIR/[video_name]/ (there are scripts that can guide how the structures are there)
    -- Masks are intially saved as images in the temp folder but when we move temp to output folder then I think we convert the masks into video with the same video metadata as input. We just remove the redundant frame from the starting of the next chunks.
    -- Stitch the (after removing redundant frames) mask videos together and then also overlay on the original video as a separate final output.

-- Simplify the README.md and also add the line that hugging face account would be needed to download checkpoints from the HF (you can check this original sam3 repo for reference btw: https://github.com/facebookresearch/sam3)

-- In the README.md also add how people can cite this work if they are using - I think we need to include 2 citations here: One for me and another one for meta research.

- Code cleanup