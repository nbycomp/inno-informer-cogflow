python3 -u run_cf_informer_local.py \
  --experiment_name small_exp_1 \
  --model informer \
  --data alibaba_pod \
  --root_path ./data/ \
  --data_path processed_data.csv \
  --features S \
  --target cpu_utilization \
  --freq 5min \
  --checkpoints ./checkpoints \
  --seq_len 12 \
  --label_len 12 \
  --pred_len 6 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --factor 5 \
  --d_model 32 \
  --n_heads 4 \
  --e_layers 1 \
  --d_layers 1 \
  --d_ff 128 \
  --dropout 0.05 \
  --attn prob \
  --embed timeF \
  --activation gelu \
  --distil \
  --padding 0 \
  --freq m \
  --batch_size 16 \
  --learning_rate 0.00001 \
  --loss mse \
  --lradj type1 \
  --num_workers 1 \
  --itr 1 \
  --train_epochs 1 \
  --patience 1 \
  --des small_exp \
  --gpu 0 \
  --devices '0'