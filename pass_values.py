import subprocess
import re
import sys
from datetime import datetime

def create_image(configurations, model_config):
    python_executable = sys.executable
    model_id = model_config["model_id"]

    # Replace non-alphanumeric characters with underscores for the filename
    model_id_with_underscores = re.sub(r'\W+', '_', model_id)
    
    # Generate the current timestamp
    current_timestamp = datetime.now().strftime(configurations['timestamp_format'])
    
    # Format the directory and filename templates with the current timestamp
    custom_directory = configurations['custom_directory']
    custom_filename = configurations['custom_filename']
    images_dir = configurations['directory_template'].format(timestamp=current_timestamp, custom_directory=custom_directory)
    filename_template = configurations['filename_template'].format(timestamp=current_timestamp, custom_filename=custom_filename)

    cmd = [python_executable, 'execution_engine.py', '--images_dir', images_dir, '--filename_template', filename_template]
    
    # Add model configuration arguments to the command
    for key, value in model_config.items():
        cmd.extend([f'--{key}', str(value)])
    
    # Loop through the other configurations and add them to the command
    for key, value in configurations.items():
        if key not in ['filename_template', 'directory_template', 'timestamp_format', 'custom_directory', 'custom_filename']:  # Already handled
            if isinstance(value, bool) and value:
                cmd.append(f'--{key}')
            elif not isinstance(value, bool):
                cmd.extend([f'--{key}', str(value)])

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Subprocess execution failed: {e}")

# Example usage with all possible configuration values provided
configurations = {
    "prompt": "A solitary house on a hill",
    "num_images": 1,
    "num_steps": 50,
    "open_image": False,
    "custom_directory": "ai_images",
    "filename_template": "{custom_filename}.png",
    "directory_template": "{custom_directory}",
    "timestamp_format": "%Y%m%d_%H%M%S",
    "add_safety_checker": True,
    "upsample_factor": 2,
    "sharpness_factor": 1.5,
    "contrast_factor": 1.3,
}

# Choose a specific model configuration
model_config = {
    "model_id": "SG161222/RealVisXL_V3.0"
}

# Replace non-alphanumeric characters with underscores for 'custom_filename'
model_id_with_underscores = re.sub(r'\W+', '_', model_config["model_id"])
configurations["custom_filename"] = model_id_with_underscores  # Update the configurations dictionary

create_image(configurations, model_config)