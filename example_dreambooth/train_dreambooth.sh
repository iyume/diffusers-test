accelerate launch --mixed_precision="fp16" dreambooth.py \
  --pretrained_model_name_or_path "stablediffusionapi/anything-v5" \
  --instance_data_dir gotou_hitori_images \
  --instance_prompt "gotou_hitori" \
  --resolution 512 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 1 --gradient_checkpointing \
  --checkpointing_steps 100 \
  --learning_rate 5e-6 \
  --lr_scheduler "constant" \
  --lr_warmup_steps 0 \
  --max_train_steps 800 \
  --enable_xformers_memory_efficient_attention \
  --set_grads_to_none
