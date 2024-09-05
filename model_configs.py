GLOBAL_IMAGE_SETTINGS = {
    "PROMPT_TO_CREATE": "tall buildings",
    "NUMBER_OF_IMAGES_TO_CREATE": 1,
    "NUMBER_OF_LOOPS": 1, 
    "NUM_INFERENCE_STEPS": 50,
    "OPEN_IMAGE_AFTER_CREATION": False,
    "IMAGES_DIRECTORY": "created_images",
    "FILENAME_TEMPLATE": "{model_prefix}_{description}_{timestamp}.png",
    "TIMESTAMP_FORMAT": "%Y%m%d_%H%M%S_%f",
    "ADD_SAFETY_CHECKER": False,
    "DEFAULT_MODEL_ID": "stablediffusionapi/disney-pixar-cartoon",
    "GLOBAL_NEGATIVE_PROMPT": "blurry, extra limbs, extra fingers, crossed eyes",
    "GLOBAL_GUIDANCE_VALUE": 7.5,
    "LOG_FILENAME": "generation_log.csv",
    "GLOBAL_LORA_ENABLED": True,
    "OPEN_IMAGE_AFTER_CREATION": False
}

PILLOW_CONFIG = {
    "UPSAMPLE_FACTOR": 2,
    "SHARPNESS_ENHANCEMENT_FACTOR": 2.0,
    "CONTRAST_ENHANCEMENT_FACTOR": 1.5,
}

MODEL_CONFIGS = {
    "DreamShaper-XL-v2-turbo": {
        "MODEL_ID": "Lykon/dreamshaper-xl-v2-turbo",
        "MODEL_PREFIX": "dreamshaperXLv2",
    },
    "RealVisXL_V4.0": {
        "MODEL_ID": "SG161222/RealVisXL_V4.0",
        "MODEL_PREFIX": "realVisXLv4",
    }
}

GLOBAL_LORA_MODEL_LIST = [
    {"repo": "nerijs/pixel-art-xl", "weight_name": "pixel-art-xl.safetensors", "adapter_name": "pixel", "trigger_phrase": "pixel art", "strength": 0.1},
    {"repo": "CiroN2022/toy-face", "weight_name": "toy_face_sdxl.safetensors", "adapter_name": "toy", "trigger_phrase": "toy_face", "strength": 0.2},
    {"repo": "Blib-la/caricature_lora_sdxl", "weight_name": "caricature_sdxl_v2.safetensors", "adapter_name": "caricature", "trigger_phrase": "caricature", "strength": 2.4},
    {"repo": "ntc-ai/SDXL-LoRA-slider.nice-hands", "weight_name": "nice hands.safetensors", "adapter_name": "nice hands", "trigger_phrase": "nice hands", "strength": 0.5},
    {"repo": "ntc-ai/SDXL-LoRA-slider.huge-anime-eyes", "weight_name": "huge anime eyes.safetensors", "adapter_name": "huge anime eyes", "trigger_phrase": "huge anime eyes", "strength": 1.75},
    {"repo": "ntc-ai/SDXL-LoRA-slider.micro-details-fine-details-detailed", "weight_name": "micro details, fine details, detailed.safetensors", "adapter_name": "micro", "trigger_phrase": "detailed", "strength": 2.4},
    {"repo": "ntc-ai/SDXL-LoRA-slider.radiant-green-eyes", "weight_name": "radiant green eyes.safetensors", "adapter_name": "radiant green eyes", "trigger_phrase": "radiant green eyes", "strength": 1.2},
    {"repo": "ntc-ai/SDXL-LoRA-slider.Studio-Ghibli-style", "weight_name": "Studio Ghibli style.safetensors", "adapter_name": "Studio Ghibli style", "trigger_phrase": "Studio Ghibli style", "strength": 3.0},
    {"repo": "ntc-ai/SDXL-LoRA-slider.extremely-detailed", "weight_name": "extremely detailed.safetensors", "adapter_name": "extremely detailed", "trigger_phrase": "extremely detailed", "strength": 1.0},
]


for config in MODEL_CONFIGS.values():
    config.update(GLOBAL_IMAGE_SETTINGS)
    config.update(PILLOW_CONFIG)

REQUIRED_PACKAGES = [
    "pillow",
    "diffusers",
    "transformers",
    "accelerate",
    "safetensors",
]