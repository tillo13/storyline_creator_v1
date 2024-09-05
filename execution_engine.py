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
from model_configs import GLOBAL_LORA_MODEL_LIST
from compel import Compel, ReturnedEmbeddingsType
import argparse

import nltk
nltk.download('punkt')

# GLOBAL LORA SETTINGS
from diffusers import DiffusionPipeline


def prepend_lora_trigger_phrases(prompt, selected_loras):
    if selected_loras:
        trigger_phrases = [model['trigger_phrase'] for model in selected_loras]
        # Prepend selected trigger phrases to the prompt
        prompt_with_triggers = ', '.join(trigger_phrases) + ', ' + prompt
        return prompt_with_triggers
    else:
        return prompt


    
def create_pipe_with_lora(model_id: str, device: str, selected_loras: list):
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

    if selected_loras:
        for lora_model in selected_loras:
            print(f"Loading LoRA model: {lora_model['repo']}")
            # Use the information from lora_model to load LoRA weights
            pipe.load_lora_weights(
                lora_model["repo"],
                weight_name=lora_model["weight_name"],
                adapter_name=lora_model["adapter_name"],
                scale=lora_model.get("strength", 1.0)  # Default strength to 1.0 if not specified
            )
    else:
        print("No LoRA models selected. Proceeding without loading LoRA models.")

    return pipe


# Determine the OS and set cache path accordingly
if platform.system() == "Windows":
    # On Windows, use %LOCALAPPDATA% for cache directory
    HF_CACHE_HOME = os.path.expanduser(os.getenv("HF_HOME", os.path.join(os.getenv("LOCALAPPDATA"), "huggingface")))
else:
    # On Unix-like systems (Linux/macOS), default to XDG cache home or ~/.cache
    HF_CACHE_HOME = os.path.expanduser(os.getenv("HF_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "huggingface")))

DEFAULT_CACHE_PATH = os.path.join(HF_CACHE_HOME, "diffusers")
# Determine cache paths
DIFFUSERS_CACHE_PATH = os.path.join(HF_CACHE_HOME, "diffusers")
HUB_CACHE_PATH = os.path.join(HF_CACHE_HOME, "hub")

def get_user_selected_config():
    # This function is only called when no arguments are passed, i.e., when run interactively
    return list_models_and_choose()


def generate_with_long_prompt(pipe, cfg, device, modified_prompt, selected_loras):
    # Fetch and modify the prompt based on selected LoRAs
    modified_prompt = prepend_lora_trigger_phrases(cfg["PROMPT_TO_CREATE"], selected_loras)

    print(f"Processing long prompt of length {len(modified_prompt)}")
    # If the tokenizer attribute exists, use it for token calculations
    #####THIS BLOCK IS TO COUNT TOKENS IS ALL####
    if hasattr(pipe, "tokenizer"):
        tokenizer = pipe.tokenizer
    elif hasattr(pipe, 'text_model') and hasattr(pipe.text_model, 'tokenizer'):
        tokenizer = pipe.text_model.tokenizer
    else:
        raise AttributeError("Tokenizer not found within the provided 'pipe' object.")

    tokenized_prompt = tokenizer.encode(modified_prompt)
    total_original_tokens = len(tokenized_prompt)

    print(f"Processing long prompt of length {len(modified_prompt)} characters.")
    print(f"Total original tokens in prompt: {total_original_tokens}")

    max_token_limit = 77  # Assuming a token limit, adjust as necessary for your use case.
    excess_tokens = max(0, total_original_tokens - max_token_limit)
    percentage_over = (excess_tokens / max_token_limit) * 100 if excess_tokens > 0 else 0
    
    # Print summary
    print("\n=== PROMPT LENGTH SUMMARY ===")
    if excess_tokens > 0:
        print(f"The input prompt consists of {total_original_tokens} tokens.")
        print(f"Your prompt exceeds the CLIP model's token limit by {excess_tokens} tokens.")
        print(f"Your prompt is over the limit (77) by {percentage_over:.2f}%.")
        print("We will generate your image, but trimmed off to fit 77 tokens.")
    else:
        print(f"The input prompt is within the CLIP model's token limit with {total_original_tokens} tokens.")
    print("===END OF PROMPT LENGTH SUMMARY===\n")
    #####THIS BLOCK IS TO COUNT TOKENS IS ALL####

    try:
        # Conditional initialization for Compel
        if hasattr(pipe, "tokenizer") and hasattr(pipe, "text_encoder"):
            compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
            # Display the parsed prompt
            parsed_prompt = compel.parse_prompt_string(modified_prompt)
            print(f"Parsed Prompt1: {parsed_prompt}")
        elif hasattr(pipe, 'text_model') and hasattr(pipe.text_model, 'tokenizer'):
            compel = Compel(tokenizer=pipe.text_model.tokenizer, text_encoder=pipe.text_model)
            # Display the parsed prompt
            parsed_prompt = compel.parse_prompt_string(modified_prompt)
            print(f"Parsed Prompt: {parsed_prompt}")
        else:
            raise AttributeError("The pipeline object does not have the expected tokenizer or text_encoder attributes.")

        print(f"Compel initialized with tokenizer and text encoder from the pipeline object.")
        
        # Generate the conditioning tensor
        print(f"Generating conditioning tensor with Compel...")
        conditioning = compel.build_conditioning_tensor(modified_prompt)
        print(f"Conditioning tensor shape: {conditioning.shape}")

        # Check embeddings tensor shape, should be [batch_size, sequence_length, embeddings_size]
        if conditioning.ndim == 3:
            print(f"Embeddings tensor has correct shape: {conditioning.shape}")
        else:
            print(f"WARNING: Embeddings tensor does not have the expected shape: {conditioning.shape}")

        # Generate image using the conditioning tensor
        print("Generating image with Compel generated conditioning tensor...")
        image = pipe(prompt=modified_prompt,
                    negative_prompt=model_configs.GLOBAL_IMAGE_SETTINGS["GLOBAL_NEGATIVE_PROMPT"],
                    num_inference_steps=cfg["NUM_INFERENCE_STEPS"],
                    guidance_scale=model_configs.GLOBAL_IMAGE_SETTINGS["GLOBAL_GUIDANCE_VALUE"],
                    ).images[0]

        print("Image generated successfully with Compel.")
        return image

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Falling back to default generation process.")
        image = pipe(prompt=modified_prompt,
                    negative_prompt=model_configs.GLOBAL_IMAGE_SETTINGS["GLOBAL_NEGATIVE_PROMPT"],
                    num_inference_steps=cfg["NUM_INFERENCE_STEPS"],
                    guidance_scale=model_configs.GLOBAL_IMAGE_SETTINGS["GLOBAL_GUIDANCE_VALUE"],
                    ).images[0]

        print("Image generated with default process due to exception.")
        return image

def list_cached_models():
    print("Cached models:")
    # Existing checks for diffusers cache and hub cache
    list_cache(DIFFUSERS_CACHE_PATH, "Diffusers models")
    list_cache(HUB_CACHE_PATH, "Hub models")

    # New code: Additional check for the custom Windows cache path
    if platform.system() == "Windows":
        custom_windows_cache_path = "C:\\Users\\mac\\.cache\\huggingface\\hub"
        list_cache(custom_windows_cache_path, "Custom Windows Hub models")

def list_cache(cache_path, description):
    print(f"\n{description}:")
    if not os.path.isdir(cache_path):
        print(f"No cache directory found at '{cache_path}'.")
    else:
        model_directories = [d for d in os.listdir(cache_path) if os.path.isdir(os.path.join(cache_path, d))]
        if not model_directories:
            print("No cached models found.")
        else:
            for idx, model_dir in enumerate(model_directories, 1):
                model_dir_path = os.path.join(cache_path, model_dir)
                print(f"{idx}. {model_dir} (Location: {model_dir_path})")

def list_models_and_choose():
    global_settings = model_configs.GLOBAL_IMAGE_SETTINGS
    print("Global configurations:")
    print(f"   - Prompt: {global_settings['PROMPT_TO_CREATE']}")
    print(f"   - Number of images to create: {global_settings['NUMBER_OF_IMAGES_TO_CREATE']}")
    print(f"   - Inference steps: {global_settings['NUM_INFERENCE_STEPS']}")
    print()
    
    model_keys = list(model_configs.MODEL_CONFIGS.keys())
    print("Available models and their configurations:")
    for idx, model_name in enumerate(model_keys, 1):
        model_config = model_configs.MODEL_CONFIGS[model_name]
        print(f"{idx}. {model_name}")
        if 'MODEL_ID' in model_config:
            print(f"   - Model ID: {model_config['MODEL_ID']}")
        else:
            # For refiner models, print base and refiner model IDs
            print(f"   - Base Model ID: {model_config['MODEL_ID_BASE']}")
            #print(f"   - Refiner Model ID: {model_config['MODEL_ID_REFINER']}")
        print()

    selected_config = None
    while selected_config is None:
        user_input = input("Select a model by number: ")
        try:
            model_idx = int(user_input) - 1  # Adjust for 0-based indexing
            if model_idx < 0 or model_idx >= len(model_keys):
                print("Invalid selection. Please try again.")
            else:
                selected_model_key = model_keys[model_idx]
                selected_config = model_configs.MODEL_CONFIGS[selected_model_key]
                print(f"You have selected: {selected_model_key}")

        except ValueError:
            print("Invalid input. Please enter a number.")
    
    return selected_config

def install_packages(packages):
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def format_time(seconds):
    return f"{int(seconds // 60)} minutes {seconds % 60:.2f} seconds"

def open_image(path):
    try:
        if sys.platform == "darwin":  # macOS
            subprocess.run(["open", path], check=True)
        elif sys.platform == "win32":  # Windows
            os.startfile(path)
        elif sys.platform.startswith('linux'):  # Linux
            subprocess.run(["xdg-open", path], check=True)
        else:
            print("Platform not supported for opening image.")
    except Exception as e:
        print(f"Failed to open image: {e}")

def post_process_image(image):
    config_values = model_configs.CURRENT_CONFIG
    factors = (config_values["UPSAMPLE_FACTOR"], config_values["SHARPNESS_ENHANCEMENT_FACTOR"],
               config_values["CONTRAST_ENHANCEMENT_FACTOR"])
    
    print("Resizing the image...")
    image = image.resize((image.width * factors[0], image.height * factors[0]), Image.LANCZOS)
    
    print("Enhancing image sharpness...")
    sharpness_enhancer = ImageEnhance.Sharpness(image)
    image = sharpness_enhancer.enhance(factors[1])
    
    print("Increasing image contrast...")
    contrast_enhancer = ImageEnhance.Contrast(image)
    image = contrast_enhancer.enhance(factors[2])
    
    print("Post-processing complete.")
    return image

def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        cuda_device_name = torch.cuda.get_device_name(0)
        cuda_device_memory = torch.cuda.get_device_properties(0).total_memory
        cuda_device_memory_gb = cuda_device_memory / (1024 ** 3)
        hardware_summary = {
            "Device Type": "GPU",
            "Device Name": cuda_device_name,
            "Device Memory (GB)": f"{cuda_device_memory_gb:.2f}",
            "CUDA Version": torch.version.cuda,
        }
    else:
        device = torch.device("cpu")
        cpu_threads = torch.get_num_threads()
        hardware_summary = {
            "Device Type": "CPU",
            "Available Threads": cpu_threads,
        }

    # Print PyTorch version and device information
    print(f"PyTorch version: {torch.__version__}")
    device_info = f"Using device: {hardware_summary['Device Name']} with {hardware_summary['Device Memory (GB)']} GB of GPU memory and CUDA version {hardware_summary['CUDA Version']}" if "GPU" in hardware_summary["Device Type"] else f"Using device: CPU with {hardware_summary['Available Threads']} threads"
    print(device_info)

    return device, hardware_summary


def main(model_id, config_overrides=None, selected_loras=None):
    start_time = time.time()

    if config_overrides is None:
        model_configs.CURRENT_CONFIG = get_user_selected_config()
    else:
        for key, value in config_overrides.items():
            if key in model_configs.CURRENT_CONFIG:
                model_configs.CURRENT_CONFIG[key] = value

        global_settings = model_configs.GLOBAL_IMAGE_SETTINGS
        pillow_settings = model_configs.PILLOW_CONFIG

        for key, value in config_overrides.items():
            if key in global_settings:
                global_settings[key] = value
            elif key in pillow_settings:
                pillow_settings[key] = value

    list_cached_models()
    device, hardware_summary = setup_device()

    print("\nConfiguration for image generation:")
    print(f"Model ID: {model_configs.CURRENT_CONFIG['MODEL_ID']}")
    print(f"Prompt: {model_configs.CURRENT_CONFIG['PROMPT_TO_CREATE']}")
    print(f"Number of Images: {model_configs.CURRENT_CONFIG['NUMBER_OF_IMAGES_TO_CREATE']}")
    print(f"Number of Inference Steps: {model_configs.CURRENT_CONFIG['NUM_INFERENCE_STEPS']}")
    print(f"Open Image After Creation: {model_configs.CURRENT_CONFIG['OPEN_IMAGE_AFTER_CREATION']}")
    print(f"Images Directory: {model_configs.CURRENT_CONFIG['IMAGES_DIRECTORY']}")
    print(f"Filename Template: {model_configs.CURRENT_CONFIG['FILENAME_TEMPLATE']}")
    print(f"Timestamp Format: {model_configs.CURRENT_CONFIG['TIMESTAMP_FORMAT']}")
    print(f"Add Safety Checker: {model_configs.CURRENT_CONFIG['ADD_SAFETY_CHECKER']}")
    print(f"Upsample Factor: {model_configs.CURRENT_CONFIG['UPSAMPLE_FACTOR']}")
    print(f"Sharpness Enhancement Factor: {model_configs.CURRENT_CONFIG['SHARPNESS_ENHANCEMENT_FACTOR']}")
    print(f"Contrast Enhancement Factor: {model_configs.CURRENT_CONFIG['CONTRAST_ENHANCEMENT_FACTOR']}")

    cfg = model_configs.CURRENT_CONFIG
    current_timestamp = datetime.now().strftime(model_configs.GLOBAL_IMAGE_SETTINGS["TIMESTAMP_FORMAT"])
    images_directory = cfg['IMAGES_DIRECTORY']
    os.makedirs(images_directory, exist_ok=True)
    use_refiner = "MODEL_ID_REFINER" in cfg

    if use_refiner:
        pass
    else:
        pipe = create_pipe_with_lora(cfg["MODEL_ID"], device, selected_loras)

    seed = model_configs.GLOBAL_IMAGE_SETTINGS.get("SEED")
    if seed is None:
        seed = int(time.time())
        model_configs.GLOBAL_IMAGE_SETTINGS["SEED"] = seed
    torch.manual_seed(seed)
    generation_times = []

    for i in range(model_configs.GLOBAL_IMAGE_SETTINGS["NUMBER_OF_IMAGES_TO_CREATE"]):
        start_gen_time = time.time()
        print(f"Processing image {i + 1} of {model_configs.GLOBAL_IMAGE_SETTINGS['NUMBER_OF_IMAGES_TO_CREATE']}...")

        additional_args = cfg.get("ADDITIONAL_PIPELINE_ARGS_BASE", {}) if use_refiner else cfg.get("ADDITIONAL_PIPELINE_ARGS", {})
        timestamp = datetime.now().strftime(model_configs.GLOBAL_IMAGE_SETTINGS["TIMESTAMP_FORMAT"])
        model_prefix = cfg.get("MODEL_PREFIX", "")
        base_filename = model_configs.GLOBAL_IMAGE_SETTINGS["FILENAME_TEMPLATE"].format(
            model_prefix=model_prefix,
            timestamp=timestamp
        )

        cfg = model_configs.CURRENT_CONFIG
        modified_prompt = prepend_lora_trigger_phrases(cfg["PROMPT_TO_CREATE"], selected_loras)

        if use_refiner:
            pass
        else:
            if len(modified_prompt) > 77:
                print("Prompt exceeds token limit. Engaging long prompt handling function...")
                image = generate_with_long_prompt(pipe, cfg, device, modified_prompt, selected_loras)
            else:
                print("Prompt within token limit. Proceeding with regular generation process...")
                image = pipe(prompt=modified_prompt,
                             negative_prompt=model_configs.GLOBAL_IMAGE_SETTINGS["GLOBAL_NEGATIVE_PROMPT"],
                             num_inference_steps=cfg["NUM_INFERENCE_STEPS"],
                             guidance_scale=model_configs.GLOBAL_IMAGE_SETTINGS["GLOBAL_GUIDANCE_VALUE"],
                            ).images[0]

        print("Starting post-processing with Pillow...")
        image = post_process_image(image)
        final_filename = base_filename
        final_img_path = os.path.join(images_directory, final_filename)
        image.save(final_img_path)

        gen_time = time.time() - start_gen_time
        generation_times.append(gen_time)
        print(f"Image {i + 1}/{cfg['NUMBER_OF_IMAGES_TO_CREATE']} saved in '{cfg['IMAGES_DIRECTORY']}' as {final_filename}")
        print(f"Full path: {os.path.abspath(final_img_path)}")
        print(f"Single image generation time: {format_time(gen_time)}")

        if model_configs.GLOBAL_IMAGE_SETTINGS["OPEN_IMAGE_AFTER_CREATION"]:
            open_image(final_img_path)

    total_time = time.time() - start_time
    avg_time = sum(generation_times) / len(generation_times) if generation_times else 0

    print("==== SUMMARY ====")
    print(f"Total execution time: {format_time(total_time)}")
    print(f"Average generation time per image: {format_time(avg_time)}")

    for key, val in hardware_summary.items():
        print(f"{key}: {val}")

    print("\n--- Configuration Details ---")
    config_values = model_configs.CURRENT_CONFIG
    for key, val in config_values.items():
        print(f"{key}: {val}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image generation pipeline.")
    parser.add_argument('--model_id', type=str, default=model_configs.GLOBAL_IMAGE_SETTINGS["DEFAULT_MODEL_ID"], help='Model ID to automatically select for image generation.')
    parser.add_argument('--prompt', type=str, default=model_configs.GLOBAL_IMAGE_SETTINGS["PROMPT_TO_CREATE"], help='Custom prompt for image generation.')
    parser.add_argument('--num_images', type=int, default=model_configs.GLOBAL_IMAGE_SETTINGS["NUMBER_OF_IMAGES_TO_CREATE"], help='Number of images to create.')
    parser.add_argument('--num_steps', type=int, default=model_configs.GLOBAL_IMAGE_SETTINGS["NUM_INFERENCE_STEPS"], help='Number of inference steps.')
    parser.add_argument('--open_image', action='store_true', default=model_configs.GLOBAL_IMAGE_SETTINGS["OPEN_IMAGE_AFTER_CREATION"], help='Open image after creation.')
    parser.add_argument('--images_dir', type=str, default=model_configs.GLOBAL_IMAGE_SETTINGS["IMAGES_DIRECTORY"], help='Directory to save created images.')
    parser.add_argument('--filename_template', type=str, default=model_configs.GLOBAL_IMAGE_SETTINGS["FILENAME_TEMPLATE"], help='Template for naming image files.')
    parser.add_argument('--timestamp_format', type=str, default=model_configs.GLOBAL_IMAGE_SETTINGS["TIMESTAMP_FORMAT"], help='Format for timestamps in image filenames.')
    parser.add_argument('--add_safety_checker', action='store_true', default=model_configs.GLOBAL_IMAGE_SETTINGS["ADD_SAFETY_CHECKER"], help='Add a safety checker to the image generation pipeline.')
    parser.add_argument('--upsample_factor', type=int, default=model_configs.PILLOW_CONFIG["UPSAMPLE_FACTOR"], help='Factor to upsample the image.')
    parser.add_argument('--sharpness_factor', type=float, default=model_configs.PILLOW_CONFIG["SHARPNESS_ENHANCEMENT_FACTOR"], help='Factor to enhance image sharpness.')
    parser.add_argument('--contrast_factor', type=float, default=model_configs.PILLOW_CONFIG["CONTRAST_ENHANCEMENT_FACTOR"], help='Factor to enhance image contrast.')
    parser.add_argument('--global_lora_enabled', type=lambda x: (str(x).lower() == 'true'), help='Enable or disable LoRA globally', default=True)
    parser.add_argument('--selected_loras', type=str, help='Comma-separated list of selected LoRA adapters', default='')

    args, unknown = parser.parse_known_args()
    GLOBAL_LORA_ENABLED = args.global_lora_enabled
    selected_loras = [GLOBAL_LORA_MODEL_LIST[int(idx) - 1] for idx in args.selected_loras.split(',') if idx]

    if len(sys.argv) == 1:
        main(None)
    else:
        selected_config = None
        for key, config in model_configs.MODEL_CONFIGS.items():
            if 'MODEL_ID' in config and config['MODEL_ID'] == args.model_id:
                selected_config = config
                break
            elif 'MODEL_ID_BASE' in config and config['MODEL_ID_BASE'] == args.model_id:
                selected_config = config
                break
            elif 'MODEL_ID_REFINER' in config and config['MODEL_ID_REFINER'] == args.model_id:
                selected_config = config
                break

        if selected_config is None:
            print(f"Error: Model ID '{args.model_id}' not found in configurations. Exiting.")
            sys.exit(1)

        model_configs.CURRENT_CONFIG = selected_config
        config_overrides = {
            "PROMPT_TO_CREATE": args.prompt,
            "MODEL_ID": args.model_id,
            "NUMBER_OF_IMAGES_TO_CREATE": args.num_images,
            "NUM_INFERENCE_STEPS": args.num_steps,
            "OPEN_IMAGE_AFTER_CREATION": args.open_image,
            "IMAGES_DIRECTORY": args.images_dir,
            "FILENAME_TEMPLATE": args.filename_template,
            "TIMESTAMP_FORMAT": args.timestamp_format,
            "ADD_SAFETY_CHECKER": args.add_safety_checker,
            "UPSAMPLE_FACTOR": args.upsample_factor,
            "SHARPNESS_ENHANCEMENT_FACTOR": args.sharpness_factor,
            "CONTRAST_ENHANCEMENT_FACTOR": args.contrast_factor,
        }

        main(args.model_id, config_overrides, selected_loras)