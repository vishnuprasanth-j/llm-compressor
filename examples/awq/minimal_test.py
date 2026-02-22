import torch
from compressed_tensors.offload import dispatch_model
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier

# Verify GPU is available
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Select small open model (no auth required)
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

print(f"\nLoading model: {MODEL_ID}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Select calibration dataset
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

# Use fewer samples for faster testing
NUM_CALIBRATION_SAMPLES = 32
MAX_SEQUENCE_LENGTH = 256

# Load dataset and preprocess
print(f"\nLoading dataset: {DATASET_ID}")
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)


def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }


ds = ds.map(preprocess)


def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(tokenize, remove_columns=ds.column_names)

# Configure the quantization algorithm
# AWQModifier performs smoothing only and should be stacked with
# QuantizationModifier for full quantization support.
# Use duo_scaling=True 
print("\nConfiguring AWQ + QuantizationModifier recipe...")
recipe = [
    AWQModifier(ignore=["lm_head"], targets=["Linear"], duo_scaling="both"),
    QuantizationModifier(targets=["Linear"], scheme="W4A16_ASYM", ignore=["lm_head"]),
]

# Apply algorithms
print("\nRunning oneshot quantization...")
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)


print("\n\n")
print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
    model.device
)
output = model.generate(input_ids, max_new_tokens=50)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

# Save to disk compressed
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-awq-test"
print(f"Saving model to: {SAVE_DIR}")
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

print("\nAWQ test completed successfully!")
