#! /bin/bash

SEED=$1

BASE_PATH=.
OUTPUT_DIR="${BASE_PATH}/outputs/opt/seed-${SEED}"
CKPT_NAME="qwen-2-5-opt-checkpoint"

mkdir -p ${OUTPUT_DIR}

OPTS=""
# data
OPTS+=" --train_data ${BASE_PATH}/data/dolly/train.jsonl"
OPTS+=" --val_data ${BASE_PATH}/data/dolly/valid.jsonl"
OPTS+=" --test_data ${BASE_PATH}/data/vicuna/valid.jsonl"

# training
OPTS+=" --num_train_epochs 10"
OPTS+=" --batch_size 16"
OPTS+=" --val_batch_size 32"
OPTS+=" --learning_rate 5e-4"
OPTS+=" --max_len 320"
OPTS+=" --pad_to_multiple_of 1"

# devices
OPTS+=" --teach_device cuda:2"
OPTS+=" --student_device cuda:2"

# loss
OPTS+=" --hard_label_loss_weight 0.5"
OPTS+=" --orthogonal False"
OPTS+=" --span_loss True"
OPTS+=" --der_loss True"
OPTS+=" --span_weight_pooling True"
OPTS+=" --span_loss_weight True"
OPTS+=" --p 1.0"

OPTS+=" --n_encoder_finetuned 32"
OPTS+=" --hidden_loss_weights 1"

OPTS+=" --entropy_weight True"
OPTS+=" --student_layer_mapping 28 32"
OPTS+=" --teacher_layer_mapping 25 28"
OPTS+=" --split_layer_mapping 0 1 2"
OPTS+=" --w_span_loss 2.0"

# models
OPTS+=" --teacher_embedding_dimension 3584"
OPTS+=" --output_dir ${OUTPUT_DIR}"
OPTS+=" --teacher_model VoCuc/Qwen2.5-7B-Instruct-Dolly-SFT"
OPTS+=" --teacher_tokenizer Qwen/Qwen2.5-7B-Instruct"
OPTS+=" --student_model facebook/opt-2.7b"
OPTS+=" --student_tokenizer facebook/opt-2.7b"

# hf token
# OPTS+=" --hf_token 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"

# extra arguments
OPTS+=" --seed ${SEED}"
OPTS+=" --student_model_type opt"
OPTS+=" --teacher_model_type qwen"
OPTS+=" --use_lora True"
OPTS+=" --grad_accum_steps 4"

# ==== Gọi Python ====
# python run_distill_llm.py ${OPTS} >> ${OUTPUT_DIR}/train.log 2>&1
python run_distill_llm.py ${OPTS}