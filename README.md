# AI Storyline Image Generator

TL;DR
When you run create_storyline.py, it generates AI-created images based on the scenes described in storyline.json. Each scene results in an image (or set of images) that visually represents the specified characters and settings. The generated images will be saved in the directory specified in the configuration, and the process will be logged in a file (generation_log.csv).

This repository contains a set of scripts and configurations for generating images from storylines using AI models.

## Contents

- `storyline.json`: Configuration file describing the storyline, characters, scenes, and model settings.
- `create_storyline.py`: Script to read the storyline and initiate image generation for each scene.
- `execution_engine.py`: Script responsible for the actual image generation using diffusion models.
- `model_configs.py`: Configuration file containing global settings and specific model configurations.
- `pass_values.py`: Helper script to facilitate passing user configurations dynamically.
- `requirements.txt`: List of Python packages required to run the scripts.

## Files Description

### storyline.json

This file describes the storyline settings, models, characters, and scenes.

```json
{
  "model_settings": {
    ...
  },
  "global_setting": {
    ...
  },
  "characters": {
    ...
  },
  "scenes": [
    ...
  ]
}
create_storyline.py
Script to read the storyline and initiate image generation for each scene.

import subprocess
import re
import sys
import json
import os
import csv
from datetime import datetime
import time

# Function to log each generation to a CSV file
def log_to_csv(logfile_path, model_id, prompt, custom_filename, success, error_message='', duration=0.0):
    ...

# Function to create an image using the execution engine
def create_image(configurations, model_config, shared_timestamp, logfile_path, scene_description, global_lora_enabled):
    ...

# Function to estimate token count in a string
def estimate_token_count(text):
    ...

# Function to check if the prompt exceeds token limit
def check_for_token_limit(prompt, token_limit=77):
    ...

# Function to read and process the storyline JSON
def process_storyline_json(file_path):
    ...

# Function to get combined prompt from scene and storyline settings
def get_combined_prompt(scene, storyline):
    ...

# Function to print summary of the script execution
def print_summary(image_success_count, model_id, start_time, number_of_loops):
    ...

if __name__ == "__main__":
    process_storyline_json('storyline.json')
execution_engine.py
Script responsible for the actual image generation using diffusion models.

import os
import platform
import subprocess
import sys
from datetime import datetime
import time
from PIL import Image, ImageEnhance
import torch
from diffusers import DiffusionPipeline
import model_configs

# Additional imports and settings for LoRA
from compel import Compel, ReturnedEmbeddingsType
import argparse
import nltk
nltk.download('punkt')

GLOBAL_LORA_MODEL_LIST = [
    ...
]

def prepend_lora_trigger_phrases(prompt, global_lora_enabled):
    ...

def create_pipe_with_lora(model_id: str, device: str, global_lora_enabled: bool):
    ...

def generate_with_long_prompt(pipe, cfg, device, modified_prompt):
    ...

def list_cached_models():
    ...

def list_cache(cache_path, description):
    ...

def list_models_and_choose():
    ...

def install_packages(packages):
    ...

def format_time(seconds):
    ...

def open_image(path):
    ...

def post_process_image(image):
    ...

def setup_device():
    ...

def main(model_id, config_overrides=None):
    ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image generation pipeline.")
    ...
    args, unknown = parser.parse_known_args()
    GLOBAL_LORA_ENABLED = args.global_lora_enabled
    ...
    main(args.model_id, config_overrides)
model_configs.py
Configuration file containing global settings and specific model configurations.

# Global image settings that apply to any model configuration
GLOBAL_IMAGE_SETTINGS = {
    ...
}

# Shared Pillow post-processing configuration
PILLOW_CONFIG = {
    ...
}

# Model-specific configurations stored in a dictionary
MODEL_CONFIGS = {
    ...
}

# Extend global and Pillow settings to each model configuration
for config in MODEL_CONFIGS.values():
    ...

# List of required Python packages for executing the script
REQUIRED_PACKAGES = [
    ...
]
pass_values.py
Helper script to facilitate passing user configurations dynamically.

import subprocess
import re
import sys
from datetime import datetime

def create_image(configurations, model_config):
    ...

# Example usage with all possible configuration values provided
configurations = {
    ...
}

# Choose a specific model configuration
model_config = {
    ...
}

# Replace non-alphanumeric characters with underscores for 'custom_filename'
model_id_with_underscores = re.sub(r'\W+', '_', model_config["model_id"])
configurations["custom_filename"] = model_id_with_underscores  # Update the configurations dictionary

create_image(configurations, model_config)
requirements.txt
This file lists all the dependencies required to run the scripts.

pillow
diffusers
transformers
accelerate
safetensors
compel
nltk
How to Run
Install dependencies:

pip install -r requirements.txt
Prepare storyline.json:

Ensure storyline.json is configured correctly with the desired storyline, characters, scenes, and model settings.

Generate images:

Run the create_storyline.py script to initiate the image generation process based on the storyline.

python create_storyline.py
Monitor Progress:

The script logs its progress and results in the generation_log.csv file.

Notes
Each generated image will be saved in the specified directory under the images_directory setting.
Global settings and model-specific configurations can be adjusted in model_configs.py.
To customize dynamic configurations, you can use pass_values.py.
Feel free to explore and modify the scripts to suit your needs! Happy generating!