#intensive module
export SQUAD_DIR=data/
python -m examples.run_squad_av \
    --model_type bert \
    --model_name_or_path bert-base-cased-finetuned-mrpc \
    --do_eval \
    --do_train \
    --version_2_with_negative \
    --train_file $SQUAD_DIR/train-v2.0.json \
    --predict_file $SQUAD_DIR/dev-v2.0.json \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --max_seq_length 512 \
    --doc_stride 128 \
    --max_query_length=64 \
    --per_gpu_train_batch_size=6 \
    --per_gpu_eval_batch_size=8 \
    --warmup_steps=814 \
    --output_dir squad/av_bert_cased_not_lower_case_lr2e-5_len512_bs48_ep2_wm814_av_ce_fp16 \
    --eval_all_checkpoints \
    --save_steps 2500 \
    --n_best_size=20 \
    --max_answer_length=30 \
    --fp16
