#sketchy module
export DATA_DIR=./group_17_global_group_Manupatra_MRC
export TASK_NAME=squad
python -m examples.run_cls \
    --model_type xlnet \
    --model_name_or_path squad/cls_xlnet_cased \
    --task_name $TASK_NAME \
    --do_predict \
    --predict_file  $DATA_DIR/dev-v2.0.json \
    --max_seq_length 512 \
    --per_gpu_eval_batch_size=8   \
    --output_dir $DATA_DIR \
    --fp16
