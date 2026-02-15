import os
import torch
import time
from threading import Thread
from queue import Queue

import sam3
from PIL import Image
from sam3.profiler import profile
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

sam3_root = os.path.join(os.path.dirname(sam3.__file__))

Q = Queue()

@profile()
def build_model(bpe_path, device):
    # bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
    print(f"Loading model on device: {device}")
    model = build_sam3_image_model(bpe_path=bpe_path, device=device)
    return model

@profile()
def inference(model, image):
    # image_path = f"{sam3_root}/../assets/images/test_image.png"
    # image = Image.open(image_path)
    # width, height = image.size
    processor = Sam3Processor(model, confidence_threshold=0.5)
    inference_state = processor.set_image(image)
    return processor, inference_state

@profile()
def prompt_and_predict(processor, inference_state, prompt="people"):
    processor.reset_all_prompts(inference_state)
    inference_state = processor.set_text_prompt(state=inference_state, prompt=prompt)
    return inference_state

@profile()
def run_image_prompts(model, image_path, prompts): 
    image = Image.open(image_path)
    processor, inference_state = inference(model, image)
    for prompt in prompts: 
        inference_state = prompt_and_predict(processor, inference_state, prompt=prompt)

class Consumer(Thread): 
    def __init__(self, model, prompts): 
        super().__init__() 
        self.model = model
        self.prompts = prompts

    def run(self):
        global Q
        while True: 
            image_path = Q.get() 
            if image_path is None: 
                break
            run_image_prompts(self.model, image_path, self.prompts)

def test_sequential(image_folder, image_files, model, prompts):
    start = time.perf_counter()
    for image_file in image_files: 
        image_path = os.path.join(image_folder, image_file)
        run_image_prompts(model, image_path, prompts)

    end = time.perf_counter()
    print(f"[Test Sequential] Total time taken: {end - start} seconds")

def test_per_image_multithreaded(image_folder, image_files, model, prompts):
    start = time.perf_counter()
    threads = []
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file) 
        thread = Thread(target=run_image_prompts, args=(model, image_path, prompts)) 
        thread.start() 
        threads.append(thread)

    for thread in threads: 
        thread.join()

    end = time.perf_counter()
    print(f"[Test Per-Image Multithreaded] Total time taken: {end - start} seconds")

def test_fixed_multithreaded(image_folder, image_files, model, prompts, num_workers=2):
    start = time.perf_counter()
    global Q
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        Q.put(image_path)
        
    for _ in range(num_workers):
        Q.put(None) # Sentinel values to signal consumers to exit
    
    workers = []
    for _ in range(num_workers):
        worker = Consumer(model, prompts)
        workers.append(worker)
        worker.start()

    for worker in workers:
        worker.join()

    end = time.perf_counter()
    print(f"[Test Fixed Multithreaded] Total time taken: {end - start} seconds")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        print("Running on CPU. For better performance, please run on a GPU.")
        torch.backends.cpu.get_cpu_capability()
    else:
        # turn on tfloat32 for Ampere GPUs
        # https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # use bfloat16for faster training on Ampere GPUs
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    # Load Model
    bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
    model = build_model(bpe_path, device)

    # Prompts
    prompts = ["foot","people","road","floor","wall","ceiling", "table","railing","truck","drain","light","car","seat","bag","food","hair-band","pant","hair","door","window","kids"]
    
    image_folder = f"{sam3_root}/../assets/images/"
    image_files = os.listdir(image_folder)

    # Test sequential processing
    test_sequential(image_folder, image_files, model, prompts)

    # Test per-image multithreading
    test_per_image_multithreaded(image_folder, image_files, model, prompts)

    # Test fixed multithreading with a queue
    test_fixed_multithreaded(image_folder, image_files, model, prompts, num_workers=2)

    