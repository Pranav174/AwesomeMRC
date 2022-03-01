#intensive module
export SQUAD_DIR=data/
python -m examples.run_squad_av \
    --model_type bert \
    --model_name_or_path  squad/av_bert_cased_not_lower_case_lr2e-5_len512_bs48_ep2_wm814_av_ce_fp16 \
    --do_eval \
    --version_2_with_negative \
    --predict_file /home/azureuser/group_17_global_group_Manupatra_MRC/dev-v2.0.json \
    --max_seq_length 512 \
    --doc_stride 128 \
    --max_query_length=64 \
    --per_gpu_eval_batch_size=8 \
    --output_dir /home/azureuser/group_17_global_group_Manupatra_MRC \
    --n_best_size=20 \
    --max_answer_length=30 \
    --fp16
