python lora.py \
  --pretrained_model_name_or_path "stablediffusionapi/anything-v5" \
  --train_data_dir hf_datasets_hitori --image_column image --caption_column additional_feature \
  --resolution 512 --random_flip \
  --train_batch_size 1 \
  --max_train_steps 10 --checkpointing_steps 100 \
  --learning_rate 1e-04 --lr_scheduler "constant" --lr_warmup_steps 0
