import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
from PIL import Image
import glob

# Initialize the model
llm = LLM(
    model="deepseek-ai/DeepSeek-OCR",
    gpu_memory_utilization=0.6,
    enable_prefix_caching=False,
    mm_processor_cache_gb=0,
    logits_processors=[NGramPerReqLogitsProcessor],
)

# Input folders and their corresponding output folders
folder_pairs = [
    (
        "/data/gpfs/projects/punim2198/hkanagalinga/DOCUMENT_HAYSTACK/DeepSeek-OCR/ORIGNAL_DATA/invoiceVQA_500_sani",
        "/data/gpfs/projects/punim2198/hkanagalinga/DOCUMENT_HAYSTACK/DeepSeek-OCR/ORIGNAL_DATA/invoiceVQA_500_txt_deepseek",
    ),
    (
        "/data/gpfs/projects/punim2198/hkanagalinga/DOCUMENT_HAYSTACK/DeepSeek-OCR/ORIGNAL_DATA/invoiceVQA_1000_sani",
        "/data/gpfs/projects/punim2198/hkanagalinga/DOCUMENT_HAYSTACK/DeepSeek-OCR/ORIGNAL_DATA/invoiceVQA_1000_txt_deepseek",
    ),
    (
        "/data/gpfs/projects/punim2198/hkanagalinga/DOCUMENT_HAYSTACK/DeepSeek-OCR/ORIGNAL_DATA/invoiceVQA_1500_sani",
        "/data/gpfs/projects/punim2198/hkanagalinga/DOCUMENT_HAYSTACK/DeepSeek-OCR/ORIGNAL_DATA/invoiceVQA_1500_txt_deepseek",
    ),
]

# Sampling parameters
sampling_param = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    extra_args=dict(
        ngram_size=30,
        window_size=90,
        whitelist_token_ids={128821, 128822},
    ),
    skip_special_tokens=False,
)

prompt = "<image>\nFree OCR."

image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']

for input_dir, output_dir in folder_pairs:
    print(f"\n{'='*60}")
    print(f"Processing folder: {input_dir}")
    print(f"Output folder:     {output_dir}")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))

    print(f"Found {len(image_files)} images")

    processed_count = 0
    skipped_count = 0

    for idx, image_path in enumerate(image_files, 1):
        image_name = os.path.basename(image_path)
        output_filename = os.path.splitext(image_name)[0] + ".txt"
        output_path = os.path.join(output_dir, output_filename)

        if os.path.exists(output_path):
            skipped_count += 1
            print(f"Skipping {idx}/{len(image_files)}: {image_name} (already processed)")
            continue

        print(f"Processing {idx}/{len(image_files)}: {image_name}")

        image = Image.open(image_path).convert("RGB")

        model_input = [
            {"prompt": prompt, "multi_modal_data": {"image": image}},
        ]

        outputs = llm.generate(model_input, sampling_param)
        extracted_text = outputs[0].outputs[0].text

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(extracted_text)

        processed_count += 1
        print(f"  Saved: {output_filename}")

    print(f"\nFolder summary:")
    print(f"  - Newly processed: {processed_count}")
    print(f"  - Skipped (already existed): {skipped_count}")
    print(f"  - Total images: {len(image_files)}")

print(f"\n{'='*60}")
print("All folders complete!")
