# official use 7 images, we have 14 images so set --repeats half to 50, 9 images for 80 repeats
python ti_🤣👉.py --pretrained_model_name_or_path stablediffusionapi/anything-v5 --train_data_dir train_images --placeholder_token 🤣👉 --initializer_token girl --enable_xformers_memory_efficient_attention --repeats 80 \
  --num_vectors 32 --learnable_property thatisit \
  --learning_rate 5e-4 --lr_warmup_steps 0 --train_batch_size 2 \
  --save_steps 100 --resolution 256 --max_train_steps 1000 --checkpointing_steps 500
