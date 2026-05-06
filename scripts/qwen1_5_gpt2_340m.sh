#! /bin/bash
unset HF_TOKEN
unset HUGGINGFACE_HUB_TOKEN
unset HF_HUB_TOKEN
SEED=$1

# ==== Định nghĩa các biến ====
BASE_PATH=.
OUTPUT_DIR="${BASE_PATH}/outputs/gpt2_120m/seed-${SEED}"
CKPT_NAME="mistral-tiny-llama-checkpoint"

mkdir -p ${OUTPUT_DIR}

# Gom tham số vào OPTS
OPTS=""
# data
OPTS+=" --train_data ${BASE_PATH}/data/dolly/train.jsonl"
OPTS+=" --val_data ${BASE_PATH}/data/dolly/valid.jsonl"
OPTS+=" --test_data ${BASE_PATH}/data/vicuna/valid.jsonl"

# training
OPTS+=" --num_train_epochs 15"
OPTS+=" --batch_size 16"
OPTS+=" --val_batch_size 32"
OPTS+=" --learning_rate 5e-4"
OPTS+=" --max_len 320"
OPTS+=" --pad_to_multiple_of 1"

# devices
OPTS+=" --teach_device cuda:0"
OPTS+=" --student_device cuda:1"

# loss
OPTS+=" --hard_label_loss_weight 0.5"
OPTS+=" --orthogonal False"
OPTS+=" --span_loss True"
OPTS+=" --der_loss True"
OPTS+=" --span_weight_pooling True"
OPTS+=" --span_loss_weight True"
OPTS+=" --p 1.0"

OPTS+=" --teacher_layers_mapping 23 24"
OPTS+=" --student_encoder_layers_finetuned 22 23"
OPTS+=" --n_encoder_finetuned 14"
OPTS+=" --hidden_loss_weights 1 1"
OPTS+=" --output_attentions True"


# models
OPTS+=" --teacher_embedding_dimension 2048"
OPTS+=" --output_dir ${OUTPUT_DIR}"
OPTS+=" --teacher_model Qwen/Qwen1.5-1.8B"
OPTS+=" --teacher_tokenizer Qwen/Qwen1.5-1.8B"
OPTS+=" --student_model openai-community/gpt2-medium"
OPTS+=" --student_tokenizer openai-community/gpt2-medium"

# hf token
echo $HF_TOKEN
export HF_TOKEN="hf_"
OPTS+=" --hf_token $HF_TOKEN"
# extra arguments
OPTS+=" --seed ${SEED}"
# OPTS+=" --teacher_sft "
OPTS+=" --student_model_type gpt2"
OPTS+=" --teacher_model_type qwen"
OPTS+=" --use_lora True"
OPTS+=" --grad_accum_steps 4"

# ==== Gọi Python ====
python run_distill_llm.py ${OPTS} >> ${OUTPUT_DIR}/train.log 2>&1
