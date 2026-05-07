#! /bin/bash

SEED=$1

# ==== Định nghĩa các biến ====
BASE_PATH=.
OUTPUT_DIR="${BASE_PATH}/outputs/qwen1_5_gpt2_340m/seed-${SEED}"
CKPT_NAME="qwen1_5_gpt2_340m-checkpoint"

mkdir -p ${OUTPUT_DIR}

# Gom tham số vào OPTS
OPTS=""
# data
OPTS+=" --train_data ${BASE_PATH}/llm_data/dolly/train.jsonl"
OPTS+=" --val_data ${BASE_PATH}/llm_data/dolly/valid.jsonl"
OPTS+=" --test_data ${BASE_PATH}/llm_data/vicuna/valid.jsonl"

# training
OPTS+=" --num_train_epochs 15"
OPTS+=" --batch_size 4"
OPTS+=" --val_batch_size 32"
OPTS+=" --learning_rate 5e-4"
OPTS+=" --max_len 320"
OPTS+=" --pad_to_multiple_of 1"

# devices
OPTS+=" --teach_device cuda:3"
OPTS+=" --student_device cuda:3"

# loss
OPTS+=" --hard_label_loss_weight 0.5"
OPTS+=" --orthogonal False"
OPTS+=" --span_loss True"
OPTS+=" --der_loss True"
OPTS+=" --span_weight_pooling True"
OPTS+=" --span_loss_weight True"
OPTS+=" --p 1.0"

OPTS+=" --n_encoder_finetuned 24"
OPTS+=" --hidden_loss_weights 1"

OPTS+=" --entropy_weight True"
OPTS+=" --student_layer_mapping 23 24"
OPTS+=" --teacher_layer_mapping 23 24"
OPTS+=" --split_layer_mapping 0 1 2"
OPTS+=" --w_span_loss 2.0"

# models
OPTS+=" --teacher_embedding_dimension 2048"
OPTS+=" --output_dir ${OUTPUT_DIR}"
OPTS+=" --teacher_model VoCuc/Qwen1.5_1.8B_SFT_Dolly"
OPTS+=" --teacher_tokenizer VoCuc/Qwen1.5_1.8B_SFT_Dolly"
OPTS+=" --student_model openai-community/gpt2-medium"
OPTS+=" --student_tokenizer openai-community/gpt2-medium"

# hf token
OPTS+=" --hf_token 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"

# extra arguments
OPTS+=" --seed ${SEED}"
OPTS+=" --student_model_type gpt2"
OPTS+=" --teacher_model_type qwen"
OPTS+=" --use_lora True"
OPTS+=" --grad_accum_steps 4"

# ==== Gọi Python ====
python run_distill_llm.py ${OPTS} >> ${OUTPUT_DIR}/train.log 2>&1
# python run_distill_llm.py ${OPTS}