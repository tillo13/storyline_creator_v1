import subprocess
import re
import sys
import json
import os
import csv
import threading
from datetime import datetime
import time
from model_configs import MODEL_CONFIGS, GLOBAL_IMAGE_SETTINGS, GLOBAL_LORA_MODEL_LIST

def log_to_csv(logfile_path, model_id, prompt, custom_filename, success, error_message='', duration=0.0):
    current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_exists = os.path.isfile(logfile_path)
    with open(logfile_path, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['timestamp', 'model_id', 'prompt', 'custom_filename', 'seconds_to_complete', 'success', 'error_message']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'timestamp': current_timestamp,
            'model_id': model_id,
            'prompt': prompt,
            'custom_filename': custom_filename,
            'seconds_to_complete': round(duration, 2),
            'success': success,
            'error_message': error_message
        })

def estimate_token_count(text):
    tokens = text.split()
    punctuation_adjustment = sum(text.count(punc) for punc in ['.', ',', ';', ':', '!', '?', '(', ')'])
    conjunctions_adjustment = sum(tokens.count(conj) for conj in ['and', 'with', 'of'])
    return len(tokens) + punctuation_adjustment + conjunctions_adjustment

def check_for_token_limit(prompt, token_limit=77):
    estimated_tokens = estimate_token_count(prompt)
    alert_message = "\n==== YOUR PROMPT SIZE ALERT ====\n"
    if estimated_tokens > token_limit:
        alert_message += (f"Your prompt is estimated at {estimated_tokens} tokens, "
                          f"which exceeds the {token_limit} token limit. It might be truncated.\n")
    else:
        alert_message += (f"Your prompt is estimated at {estimated_tokens} tokens, "
                          f"which is within the {token_limit} token limit.\n")
    alert_message += "==== END ALERT ====\n"
    print(alert_message)

def process_storyline_json(file_path):
    start_time = time.time()
    with open(file_path, 'r') as file:
        storyline = json.load(file)
    
    default_settings = storyline["model_settings"]["default_model_settings"]
    number_of_loops = GLOBAL_IMAGE_SETTINGS.get("NUMBER_OF_LOOPS", 1)
    global_lora_enabled = GLOBAL_IMAGE_SETTINGS["GLOBAL_LORA_ENABLED"]
    open_image_after_creation = GLOBAL_IMAGE_SETTINGS["OPEN_IMAGE_AFTER_CREATION"]

    print(f"GLOBAL_LORA_ENABLED (from model configs): {global_lora_enabled}")
    print(f"OPEN_IMAGE_AFTER_CREATION (from model configs): {open_image_after_creation}")

    model_config = select_model_config()
    if not model_config:
        print(f"Error: Model configuration not found. Exiting.")
        return

    logfile_path = os.path.join(os.getcwd(), model_config["LOG_FILENAME"])
    num_images_to_create = GLOBAL_IMAGE_SETTINGS['NUMBER_OF_IMAGES_TO_CREATE']
    num_scenes = len(storyline["scenes"])
    total_images = number_of_loops * num_images_to_create * num_scenes

    estimated_initial_time = total_images * 60
    hours, remainder = divmod(estimated_initial_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"***INITIAL ESTIMATED TIME TO COMPLETE ({number_of_loops} loops, {num_scenes} scenes, {num_images_to_create} images per scene): {int(hours)}hrs, {int(minutes)} mins, {int(seconds)} seconds***")

    images_generated = 0
    image_success_count = 0
    total_duration = 0

    selected_loras = select_lora_configs(global_lora_enabled)

    for _ in range(number_of_loops):
        for scene in storyline["scenes"]:
            combination_details = get_combined_prompt(scene, storyline)
            configurations = {
                'prompt': combination_details,
                'num_images': num_images_to_create,
                'timestamp_format': GLOBAL_IMAGE_SETTINGS['TIMESTAMP_FORMAT'],
                'add_safety_checker': GLOBAL_IMAGE_SETTINGS['ADD_SAFETY_CHECKER'],
                'open_image': GLOBAL_IMAGE_SETTINGS['OPEN_IMAGE_AFTER_CREATION']
            }
            current_timestamp = datetime.now().strftime(model_config['TIMESTAMP_FORMAT'])
            scene_description = scene['description']

            for i in range(num_images_to_create):
                images_left = total_images - images_generated
                try:
                    custom_filename_suffix = f"{i+1}_of_{num_images_to_create}"
                    duration = create_image(configurations, model_config, current_timestamp, logfile_path, scene_description, selected_loras, custom_filename_suffix)
                    image_success_count += 1
                    total_duration += duration
                    images_generated += 1

                    avg_time_per_image = total_duration / images_generated
                    est_time_remaining = avg_time_per_image * images_left
                    hours, remainder = divmod(est_time_remaining, 3600)
                    minutes, seconds = divmod(remainder, 60)

                    if images_generated == 1:
                        print(f"###UPDATED ESTIMATED TIME AFTER FIRST IMAGE: {int(hours)}hrs, {int(minutes)} mins, {int(seconds)} seconds###")
                    else:
                        print(f"########################################")
                        print(f"###Estimated time remaining: {int(hours)}hrs, {int(minutes)} mins, {int(seconds)} seconds###")
                        print(f"###Images left: {images_left} ###")
                        print(f"########################################")
                except subprocess.CalledProcessError as e:
                    print(f"Image creation failed: {e}")

    print_summary(image_success_count, start_time)
    
def create_image(configurations, model_config, shared_timestamp, logfile_path, scene_description, selected_loras, custom_filename_suffix):
    start_time = time.time()
    model_id_clean = re.sub(r'\W+', '_', model_config["MODEL_ID"]).lower()
    description_slug = re.sub(r'[\W_]+', '_', scene_description)[:30].lower()
    images_dir = os.path.join(os.getcwd(), model_config["IMAGES_DIRECTORY"])
    os.makedirs(images_dir, exist_ok=True)
    
    filename_without_extension = "{}_{}_{}_{}".format(model_id_clean, description_slug, shared_timestamp.lower(), custom_filename_suffix).lower()
    full_filename = "{}.png".format(filename_without_extension)
    full_file_path = os.path.join(images_dir, full_filename)

    python_executable = sys.executable
    cmd = [
        python_executable, 'execution_engine.py', '--model_id', model_config["MODEL_ID"],
        '--prompt', configurations['prompt'], '--num_images', '1', '--num_steps', str(model_config['NUM_INFERENCE_STEPS']),
        '--images_dir', images_dir, '--filename_template', full_file_path, '--timestamp_format', configurations['timestamp_format'],
        '--global_lora_enabled', str(bool(selected_loras)).lower(),
        '--selected_loras', ','.join(str(GLOBAL_LORA_MODEL_LIST.index(lora) + 1) for lora in selected_loras)
    ]

    if configurations.get('add_safety_checker', False):
        cmd.append('--add_safety_checker')
    if configurations.get('open_image', False):
        cmd.append('--open_image')

    log_to_csv(logfile_path, model_config["MODEL_ID"], configurations['prompt'], full_filename, success=True, error_message='Attempt')

    try:
        subprocess.run(cmd, check=True)
        end_time = time.time()
        duration = end_time - start_time
        log_to_csv(logfile_path, model_config["MODEL_ID"], configurations['prompt'], full_filename, success=True, error_message='', duration=duration)
        return duration
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time
        log_to_csv(logfile_path, model_config["MODEL_ID"], configurations['prompt'], full_filename, success=False, error_message=str(e), duration=duration)
        return duration

def get_combined_prompt(scene, storyline):
    characters_summary = []
    for character_id in scene.get("focus", []):
        character = storyline["characters"].get(character_id)
        if character:
            age = character['age']
            skin_tone = character['skin_tone']
            skin_tone = skin_tone.split()[0] if skin_tone.split() else skin_tone
            key_traits = [trait for trait in [character.get('characteristics')] if trait]
            char_short_desc = f"{character['gender']}[age:{age},skin:{skin_tone},{','.join(key_traits)}]"
            characters_summary.append(char_short_desc)
    
    characters_combined = ','.join(characters_summary).replace(" ,", ",")
    prompt = (
        f"Scene '{scene['description']}' depicts,"
        f"with {len(characters_summary)} characters ({characters_combined}),"
        f"engaged in the scene, not dominating the frame. Include elements of "
        f"'{storyline['global_setting']['image_enhancements']}' in the background, "
        f"emphasizing interaction and environment."
    )
    return prompt

def get_character_description(focus, characters):
    character_descriptions = []
    for character_id in focus:
        char_info = characters.get(character_id)
        if char_info:
            description = f"{character_id.capitalize()} ({', '.join(filter(None, [char_info.get('gender'), char_info.get('age'), char_info.get('skin_tone'), char_info.get('characteristics')]))})"
            character_descriptions.append(description)
    return character_descriptions

def print_summary(image_success_count, start_time):
    total_time_taken = time.time() - start_time
    hours, remainder = divmod(total_time_taken, 3600)
    minutes, seconds = divmod(remainder, 60)
    print("\n===SUMMARY===")
    print(f"Time to run: {int(hours)}hrs, {int(minutes)} mins, {int(seconds)} seconds")
    print(f"Images created: {image_success_count}")

def select_model_config():
    model_keys = list(MODEL_CONFIGS.keys())
    print("Please select a model to use by entering the number:")
    for idx, model_name in enumerate(model_keys, 1):
        print(f"{idx}. {model_name}")

    selected_config = [None]

    def get_user_input():
        try:
            user_input = input("Enter the number of the model you want to use: ")
            model_idx = int(user_input) - 1
            if 0 <= model_idx < len(model_keys):
                selected_config[0] = MODEL_CONFIGS[model_keys[model_idx]]
            else:
                print("Invalid selection. Defaulting to the last model in the list.")
                selected_config[0] = MODEL_CONFIGS[model_keys[-1]]
        except Exception:
            print("Invalid input or timeout. Defaulting to the last model in the list.")
            selected_config[0] = MODEL_CONFIGS[model_keys[-1]]

    user_input_thread = threading.Thread(target=get_user_input)
    user_input_thread.start()
    user_input_thread.join(timeout=10)

    if selected_config[0] is None:
        print("User did not select a model in time. Defaulting to the last model in the list.")
        selected_config[0] = MODEL_CONFIGS[model_keys[-1]]

    print(f"Selected Model: {selected_config[0]['MODEL_ID']}")
    return selected_config[0]



def select_lora_configs(global_lora_enabled):
    if not global_lora_enabled:
        return []

    print("Please select which LoRAs to use by entering the numbers separated by commas (e.g., 1,3,5):")
    for idx, lora in enumerate(GLOBAL_LORA_MODEL_LIST, 1):
        print(f"{idx}. {lora['adapter_name']} (Trigger Phrase: {lora['trigger_phrase']}, Strength: {lora['strength']})")
    
    selected_loras = []
    user_input = input("Enter your choices: ")
    try:
        user_choices = [int(choice.strip()) - 1 for choice in user_input.split(",")]
        for choice in user_choices:
            if 0 <= choice < len(GLOBAL_LORA_MODEL_LIST):
                selected_loras.append(GLOBAL_LORA_MODEL_LIST[choice])
            else:
                print(f"Invalid choice: {choice + 1}. Ignoring it.")
    except ValueError:
        print("Invalid input. No LoRAs selected.")
    
    return selected_loras

if __name__ == "__main__":
    # Ensure the MODEL_CONFIGS contains valid model_id keys
    for model_name, config in MODEL_CONFIGS.items():
        if 'MODEL_ID' not in config:
            print(f"Invalid model configuration for {model_name}")
            sys.exit(1)
    
    process_storyline_json('storyline.json')