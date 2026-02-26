.PHONY: help setup check-uv install test test-fast test-slow test-image test-video \
        run run-all run-example profile profile-all profile-example clean clean-outputs \
        clean-cache clean-profile distclean lint format check \
        image-prompter video-prompter linkedin-visuals

# Default target
.DEFAULT_GOAL := help

# Color output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

# Project configuration
PYTHON := python3
UV := uv
VIDEO_RES := 480p
EXAMPLE_DIR := examples
TEST_DIR := tests
OUTPUT_DIR := results

# Check if uv is installed
check-uv:
	@command -v $(UV) >/dev/null 2>&1 || { \
		echo "$(RED)Error: uv is not installed!$(NC)"; \
		echo "$(YELLOW)Please run: make setup$(NC)"; \
		echo "$(YELLOW)Or manually run: ./setup.sh$(NC)"; \
		exit 1; \
	}

help: ## Show this help message
	@echo "$(BLUE)SAM3 CPU - Makefile Commands$(NC)"
	@echo ""
	@echo "$(GREEN)Setup:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; /^setup|^check-uv|^install/ {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)CLI Tools:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; /^image-prompter|^video-prompter|^linkedin/ {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Running Examples:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; /^run/ {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Profiling:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; /^profile/ {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Testing:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; /^test/ {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Maintenance:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; /^clean|^lint|^format|^check/ {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Variables:$(NC)"
	@echo "  VIDEO_RES=$(VIDEO_RES)  (set resolution: 480p, 720p, 1080p)"
	@echo ""
	@echo "$(GREEN)Examples:$(NC)"
	@echo "  make run-all VIDEO_RES=720p      # Run all examples with 720p video"
	@echo "  make run-example EXAMPLE=a       # Run specific example (a-i)"
	@echo "  make profile-all                 # Run all with profiling enabled"
	@echo "  make test-fast                   # Run only fast tests"
	@echo "  make test ARGS='-v'              # Run tests with verbose output"
	@echo "  make image-prompter IMAGES='img.jpg' PROMPTS='person car'"
	@echo "  make video-prompter VIDEO='clip.mp4' PROMPTS='player'"

setup: ## Run setup script to install uv and dependencies
	@if [ ! -f setup.sh ]; then \
		echo "$(RED)Error: setup.sh not found!$(NC)"; \
		exit 1; \
	fi
	@chmod +x setup.sh
	@./setup.sh

install: check-uv ## Install project dependencies using uv
	@echo "$(BLUE)Installing dependencies...$(NC)"
	$(UV) pip install -e .
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

test: check-uv ## Run all tests with pytest
	@echo "$(BLUE)Running all tests...$(NC)"
	$(UV) run pytest $(TEST_DIR) $(ARGS)

test-fast: check-uv ## Run only fast tests (skip slow video tests)
	@echo "$(BLUE)Running fast tests...$(NC)"
	$(UV) run pytest $(TEST_DIR) -m "not slow" $(ARGS)

test-slow: check-uv ## Run only slow tests
	@echo "$(BLUE)Running slow tests...$(NC)"
	$(UV) run pytest $(TEST_DIR) -m "slow" $(ARGS)

test-image: check-uv ## Run only image processing tests
	@echo "$(BLUE)Running image tests...$(NC)"
	$(UV) run pytest $(TEST_DIR) -m "image" $(ARGS)

test-video: check-uv ## Run only video processing tests
	@echo "$(BLUE)Running video tests...$(NC)"
	$(UV) run pytest $(TEST_DIR) -m "video" $(ARGS)

test-scenario: check-uv ## Run specific scenario test (e.g., make test-scenario SCENARIO=a)
	@if [ -z "$(SCENARIO)" ]; then \
		echo "$(RED)Error: SCENARIO not specified$(NC)"; \
		echo "$(YELLOW)Usage: make test-scenario SCENARIO=a$(NC)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Running scenario $(SCENARIO) tests...$(NC)"
	$(UV) run pytest $(TEST_DIR) -m "scenario_$(SCENARIO)" $(ARGS)

test-coverage: check-uv ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	$(UV) run pytest $(TEST_DIR) --cov=sam3 --cov-report=html --cov-report=term $(ARGS)
	@echo "$(GREEN)✓ Coverage report generated in htmlcov/index.html$(NC)"

run-all: check-uv ## Run all examples (scenarios a-i)
	@echo "$(BLUE)Running all SAM3 examples (video resolution: $(VIDEO_RES))...$(NC)"
	$(UV) run $(PYTHON) $(EXAMPLE_DIR)/run_all_examples.py \
		--video-resolution $(VIDEO_RES) \
		--output-base $(OUTPUT_DIR) \
		--continue-on-error \
		$(ARGS)

run-example: check-uv ## Run specific example (e.g., make run-example EXAMPLE=a)
	@if [ -z "$(EXAMPLE)" ]; then \
		echo "$(RED)Error: EXAMPLE not specified$(NC)"; \
		echo "$(YELLOW)Usage: make run-example EXAMPLE=a$(NC)"; \
		echo "$(YELLOW)Available examples: a, b, c, d, e, f, g, h, i$(NC)"; \
		exit 1; \
	fi
	@SCRIPT=$$(ls $(EXAMPLE_DIR)/example_$(EXAMPLE)_*.py 2>/dev/null | head -1); \
	if [ -n "$$SCRIPT" ]; then \
		echo "$(BLUE)Running example $(EXAMPLE)...$(NC)"; \
		$(UV) run $(PYTHON) $$SCRIPT --output $(OUTPUT_DIR)/example_$(EXAMPLE) $(ARGS); \
	else \
		echo "$(RED)Error: Example '$(EXAMPLE)' not found. Available examples: a, b, c, d, e, f, g, h, i$(NC)"; \
		exit 1; \
	fi

run-a: check-uv ## Run example A: single image with text prompts
	@$(UV) run $(PYTHON) $(EXAMPLE_DIR)/example_a_image_with_prompts.py $(ARGS)

run-b: check-uv ## Run example B: single image with bounding boxes
	@$(UV) run $(PYTHON) $(EXAMPLE_DIR)/example_b_image_with_boxes.py $(ARGS)

run-c: check-uv ## Run example C: batch images with text prompts
	@$(UV) run $(PYTHON) $(EXAMPLE_DIR)/example_c_batch_images_with_prompts.py $(ARGS)

run-d: check-uv ## Run example D: batch images with bounding boxes
	@$(UV) run $(PYTHON) $(EXAMPLE_DIR)/example_d_batch_images_with_boxes.py $(ARGS)

run-e: check-uv ## Run example E: video with text prompts
	@$(UV) run $(PYTHON) $(EXAMPLE_DIR)/example_e_video_with_prompts.py \
		--video assets/videos/Roger-Federer-vs-Rafael-Nadal-Wimbledon-2008_$(VIDEO_RES).mp4 $(ARGS)

run-f: check-uv ## Run example F: video with point prompts
	@$(UV) run $(PYTHON) $(EXAMPLE_DIR)/example_f_video_with_points.py \
		--video assets/videos/Roger-Federer-vs-Rafael-Nadal-Wimbledon-2008_$(VIDEO_RES).mp4 $(ARGS)

run-g: check-uv ## Run example G: refine video object
	@$(UV) run $(PYTHON) $(EXAMPLE_DIR)/example_g_refine_video_object.py \
		--video assets/videos/Roger-Federer-vs-Rafael-Nadal-Wimbledon-2008_$(VIDEO_RES).mp4 $(ARGS)

run-h: check-uv ## Run example H: remove video objects
	@$(UV) run $(PYTHON) $(EXAMPLE_DIR)/example_h_remove_video_objects.py \
		--video assets/videos/Roger-Federer-vs-Rafael-Nadal-Wimbledon-2008_$(VIDEO_RES).mp4 $(ARGS)

run-i: check-uv ## Run example I: video with segment prompts
	@$(UV) run $(PYTHON) $(EXAMPLE_DIR)/example_i_video_with_segments.py \
		--video assets/videos/Roger-Federer-vs-Rafael-Nadal-Wimbledon-2008_$(VIDEO_RES).mp4 $(ARGS)

profile-all: check-uv ## Run all examples with profiling enabled
	@echo "$(BLUE)Running all SAM3 examples with profiling...$(NC)"
	$(UV) run $(PYTHON) $(EXAMPLE_DIR)/run_all_examples.py \
		--video-resolution $(VIDEO_RES) \
		--output-base $(OUTPUT_DIR) \
		--continue-on-error \
		--profile $(ARGS)
	@echo "$(GREEN)✓ Profiling results saved to profile_results.json and profile_results.txt$(NC)"

profile-example: check-uv ## Run specific example with profiling (e.g., make profile-example EXAMPLE=a)
	@if [ -z "$(EXAMPLE)" ]; then \
		echo "$(RED)Error: EXAMPLE not specified$(NC)"; \
		echo "$(YELLOW)Usage: make profile-example EXAMPLE=a$(NC)"; \
		echo "$(YELLOW)Available examples: a, b, c, d, e, f, g, h, i$(NC)"; \
		exit 1; \
	fi
	@SCRIPT=$$(ls $(EXAMPLE_DIR)/example_$(EXAMPLE)_*.py 2>/dev/null | head -1); \
	if [ -n "$$SCRIPT" ]; then \
		echo "$(BLUE)Running example $(EXAMPLE) with profiling...$(NC)"; \
		$(UV) run $(PYTHON) $$SCRIPT --output $(OUTPUT_DIR)/example_$(EXAMPLE) --profile $(ARGS); \
		echo "$(GREEN)✓ Profiling results saved$(NC)"; \
	else \
		echo "$(RED)Error: Could not find example script$(NC)"; \
		exit 1; \
	fi

profile: profile-all ## Alias for profile-all

lint: check-uv ## Run code linting with ruff
	@echo "$(BLUE)Running linter...$(NC)"
	$(UV) run ruff check sam3/ $(ARGS)

format: check-uv ## Format code with ruff
	@echo "$(BLUE)Formatting code...$(NC)"
	$(UV) run ruff format sam3/
	@echo "$(GREEN)✓ Code formatted$(NC)"

check: check-uv lint test-fast ## Run linter and fast tests
	@echo "$(GREEN)✓ All checks passed$(NC)"

clean: ## Remove output files and __pycache__
	@echo "$(BLUE)Cleaning outputs and cache...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✓ Cleaned$(NC)"

clean-outputs: ## Remove all output directories
	@echo "$(BLUE)Cleaning output directories...$(NC)"
	rm -rf $(OUTPUT_DIR)
	@echo "$(GREEN)✓ Output directories removed$(NC)"

clean-cache: ## Remove pytest cache and coverage files
	@echo "$(BLUE)Cleaning test cache...$(NC)"
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -f .coverage
	@echo "$(GREEN)✓ Test cache cleaned$(NC)"

clean-profile: ## Remove profiling results
	@echo "$(BLUE)Cleaning profiling results...$(NC)"
	rm -f profile_results.json
	rm -f profile_results.txt
	@echo "$(GREEN)✓ Profiling results removed$(NC)"

distclean: clean clean-outputs clean-cache clean-profile ## Complete cleanup (outputs, cache, dependencies)
	@echo "$(BLUE)Complete cleanup...$(NC)"
	rm -rf .venv
	rm -rf build
	rm -rf dist
	@echo "$(GREEN)✓ Complete cleanup done$(NC)"

benchmark: check-uv ## Run performance benchmarks
	@echo "$(BLUE)Running benchmarks...$(NC)"
	@echo "$(YELLOW)Benchmark results:$(NC)"
	@for res in 480p 720p 1080p; do \
		echo "$(YELLOW)Testing $$res resolution:$(NC)"; \
		time $(UV) run $(PYTHON) $(EXAMPLE_DIR)/example_e_video_with_prompts.py \
			--video assets/videos/Roger-Federer-vs-Rafael-Nadal-Wimbledon-2008_$$res.mp4 \
			--output $(OUTPUT_DIR)/benchmark_$$res 2>&1 | grep -E "real|user|sys"; \
	done

# ---------------------------------------------------------------------------
# CLI Tools
# ---------------------------------------------------------------------------

image-prompter: check-uv ## Run image_prompter.py (IMAGES, PROMPTS, POINTS, BBOX, ALPHA, DEVICE, OUTPUT)
	@if [ -z "$(IMAGES)" ]; then \
		echo "$(RED)Error: IMAGES not specified$(NC)"; \
		echo "$(YELLOW)Usage: make image-prompter IMAGES='photo.jpg' PROMPTS='person car'$(NC)"; \
		echo "$(YELLOW)       make image-prompter IMAGES='a.jpg b.jpg' BBOX='100 50 400 300'$(NC)"; \
		echo "$(YELLOW)       make image-prompter IMAGES='img.jpg' POINTS='320,240' POINT_LABELS='1'$(NC)"; \
		exit 1; \
	fi
	@CMD="$(UV) run $(PYTHON) image_prompter.py --images $(IMAGES)"; \
	if [ -n "$(PROMPTS)" ]; then CMD="$$CMD --prompts $(PROMPTS)"; fi; \
	if [ -n "$(POINTS)" ]; then CMD="$$CMD --points $(POINTS)"; fi; \
	if [ -n "$(POINT_LABELS)" ]; then CMD="$$CMD --point-labels $(POINT_LABELS)"; fi; \
	if [ -n "$(BBOX)" ]; then CMD="$$CMD --bbox $(BBOX)"; fi; \
	if [ -n "$(ALPHA)" ]; then CMD="$$CMD --alpha $(ALPHA)"; fi; \
	if [ -n "$(DEVICE)" ]; then CMD="$$CMD --device $(DEVICE)"; fi; \
	if [ -n "$(OUTPUT)" ]; then CMD="$$CMD --output $(OUTPUT)"; else CMD="$$CMD --output $(OUTPUT_DIR)"; fi; \
	echo "$(BLUE)Running image_prompter...$(NC)"; \
	$$CMD $(ARGS)

video-prompter: check-uv ## Run video_prompter.py (VIDEO, PROMPTS, POINTS, MASKS, ALPHA, DEVICE, OUTPUT, ...)
	@if [ -z "$(VIDEO)" ]; then \
		echo "$(RED)Error: VIDEO not specified$(NC)"; \
		echo "$(YELLOW)Usage: make video-prompter VIDEO='clip.mp4' PROMPTS='person ball'$(NC)"; \
		echo "$(YELLOW)       make video-prompter VIDEO='clip.mp4' PROMPTS='player' FRAME_RANGE='100 500'$(NC)"; \
		echo "$(YELLOW)       make video-prompter VIDEO='clip.mp4' PROMPTS='player' TIME_RANGE='0:05 0:30'$(NC)"; \
		echo "$(YELLOW)       make video-prompter VIDEO='clip.mp4' MASKS='mask.png'$(NC)"; \
		exit 1; \
	fi
	@CMD="$(UV) run $(PYTHON) video_prompter.py --video $(VIDEO)"; \
	if [ -n "$(PROMPTS)" ]; then CMD="$$CMD --prompts $(PROMPTS)"; fi; \
	if [ -n "$(POINTS)" ]; then CMD="$$CMD --points $(POINTS)"; fi; \
	if [ -n "$(POINT_LABELS)" ]; then CMD="$$CMD --point-labels $(POINT_LABELS)"; fi; \
	if [ -n "$(MASKS)" ]; then CMD="$$CMD --masks $(MASKS)"; fi; \
	if [ -n "$(ALPHA)" ]; then CMD="$$CMD --alpha $(ALPHA)"; fi; \
	if [ -n "$(DEVICE)" ]; then CMD="$$CMD --device $(DEVICE)"; fi; \
	if [ -n "$(CHUNK_SPREAD)" ]; then CMD="$$CMD --chunk-spread $(CHUNK_SPREAD)"; fi; \
	if [ -n "$(FRAME_RANGE)" ]; then CMD="$$CMD --frame-range $(FRAME_RANGE)"; fi; \
	if [ -n "$(TIME_RANGE)" ]; then CMD="$$CMD --time-range $(TIME_RANGE)"; fi; \
	if [ -n "$(KEEP_TEMP)" ]; then CMD="$$CMD --keep-temp"; fi; \
	if [ -n "$(OUTPUT)" ]; then CMD="$$CMD --output $(OUTPUT)"; else CMD="$$CMD --output $(OUTPUT_DIR)"; fi; \
	echo "$(BLUE)Running video_prompter...$(NC)"; \
	$$CMD $(ARGS)

info: ## Display project information
	@echo "$(BLUE)SAM3 CPU Project Information$(NC)"
	@echo ""
	@echo "Project: SAM3 (Segment Anything Model 3) - CPU Compatible"
	@echo "OS: $$(uname -s)"
	@echo "Architecture: $$(uname -m)"
	@echo "Python: $$($(PYTHON) --version 2>&1)"
	@command -v $(UV) >/dev/null 2>&1 && echo "UV: $$($(UV) --version 2>&1)" || echo "UV: $(RED)not installed$(NC)"
	@echo "PyTorch: $$($(UV) run $(PYTHON) -c 'import torch; print(torch.__version__)' 2>/dev/null || echo '$(RED)not installed$(NC)')"
	@echo "CUDA Available: $$($(UV) run $(PYTHON) -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo '$(RED)unknown$(NC)')"
	@command -v ffmpeg >/dev/null 2>&1 && echo "FFmpeg: $$(ffmpeg -version 2>&1 | head -n1 | cut -d' ' -f3)" || echo "FFmpeg: $(RED)not installed$(NC)"
	@echo ""
	@echo "$(GREEN)Project Structure:$(NC)"
	@echo "  sam3/              - Core library"
	@echo "  examples/          - Demo scripts (9 scenarios)"
	@echo "  tests/             - Test suite"
	@echo "  assets/            - Sample images and videos"
	@echo ""
	@echo "$(GREEN)Profiling:$(NC)"
	@echo "  Use --profile flag or make profile-* targets to enable profiling"
	@echo "  Results saved to: profile_results.json, profile_results.txt"
	@echo ""
