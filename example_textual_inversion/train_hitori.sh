python textual_inversion.py --pretrained_model_name_or_path stablediffusionapi/anything-v5 --train_data_dir gotou_hitori_images --placeholder_token gotou_hitori --initializer_token girl --enable_xformers_memory_efficient_attention --repeats 60 \
  --num_vectors 32 --learnable_property hitori \
  --learning_rate 5e-4 --lr_warmup_steps 0 --train_batch_size 1 \
  --save_steps 500 --resolution 512 --max_train_steps 5000 --checkpointing_steps 500 --resume_from_checkpoint text-inversion-model/checkpoint-3000
