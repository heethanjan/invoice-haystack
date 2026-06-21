#!/bin/bash

DATASET_FOLDER="/data/gpfs/projects/punim2198/hkanagalinga/DOCUMENT_HAYSTACK/1_DATA"
DATASET_FILE="$DATASET_FOLDER/test_invoiceVQA.json"
IMAGE_ROOT="$DATASET_FOLDER/Test"
IMAGE_DIR="invoiceVQA_500"
TEXT_DIR="invoiceVQA_500_txt_deepseek"

OUTPUT_DIR="/data/gpfs/projects/punim2198/hkanagalinga/DOCUMENT_HAYSTACK/output/invoicevqa_500_test_invoice_hybrid"

python VRAG_retrieval_hybrid.py \
    --dataset_file $DATASET_FILE \
    --image_root $IMAGE_ROOT \
    --image_dir $IMAGE_DIR \
    --text_dir $TEXT_DIR \
    --output_dir $OUTPUT_DIR \
    --use_question_query \

# echo "----------------------------------------"

# echo "----------------------------------------"
# echo "----------------------------------------"
# echo "500Hybrid Retrieval Completed!"
# echo "Results saved to: $OUTPUT_DIR"
# echo "----------------------------------------" 
# #!/bin/bash
# DATASET_FOLDER="/data/projects/punim2198/hkanagalinga/DOCUMENT_HAYSTACK/dochaystacks/data/Document_Haystacks"
# DATASET_FILE="$DATASET_FOLDER/test_invoiceVQA.json"
# IMAGE_ROOT="$DATASET_FOLDER/Test"
# IMAGE_DIR="invoiceVQA_1000"
# TEXT_DIR="invoiceVQA_1000_txt_deepseek"

# OUTPUT_DIR="/data/gpfs/projects/punim2198/hkanagalinga/DOCUMENT_HAYSTACK/output/invoicevqa_1000_test_invoice_hybrid"

# python VRAG_retrieval_hybrid.py \
#     --dataset_file $DATASET_FILE \
#     --image_root $IMAGE_ROOT \
#     --image_dir $IMAGE_DIR \
#     --text_dir $TEXT_DIR \
#     --output_dir $OUTPUT_DIR \
#     --use_question_query \


# echo "----------------------------------------"

# echo "----------------------------------------"
# echo "----------------------------------------"
# echo "1000Hybrid Retrieval Completed!"
# echo "Results saved to: $OUTPUT_DIR"
# echo "----------------------------------------" 
# #!/bin/bash
# DATASET_FOLDER="/data/projects/punim2198/hkanagalinga/DOCUMENT_HAYSTACK/dochaystacks/data/Document_Haystacks"
# DATASET_FILE="$DATASET_FOLDER/test_invoiceVQA.json"
# IMAGE_ROOT="$DATASET_FOLDER/Test"
# IMAGE_DIR="invoiceVQA_1500"
# TEXT_DIR="invoiceVQA_1500_txt_deepseek"

# OUTPUT_DIR="/data/gpfs/projects/punim2198/hkanagalinga/DOCUMENT_HAYSTACK/output/invoicevqa_1500_test_invoice_hybrid"

# python VRAG_retrieval_hybrid.py \
#     --dataset_file $DATASET_FILE \
#     --image_root $IMAGE_ROOT \
#     --image_dir $IMAGE_DIR \
#     --text_dir $TEXT_DIR \
#     --output_dir $OUTPUT_DIR \
#     --use_question_query \


