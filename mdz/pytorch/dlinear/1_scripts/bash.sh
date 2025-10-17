 --task_name long_term_forecast --is_training 0 --checkpoints ../0_dlinear/checkpoints/ --root_path ../0_dlinear/dataset/ETT-small/ --data_path ETTh1.csv  --model_id ETTh1_96_96 --model DLinear --data ETTh1 --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1  --factor 3 --enc_in 7  --dec_in 7   --c_out 7 --des 'Exp' --itr 1 --batch_size 1 --model_pt ../2_compile/fmodel/dlinear_ltf.pt


 --task_name classification --is_training 0 --checkpoints ../0_dlinear/checkpoints/ --root_path ../0_dlinear/dataset/Heartbeat/ --model_id Heartbeat --model DLinear --data UEA --e_layers 3 --batch_size 1 --d_model 128 --d_ff 256 --top_k 3 --des 'Exp' --itr 1 --learning_rate 0.001 --train_epochs 100 --patience 10  --model_pt ../2_compile/fmodel/dlinear_cls.pt

