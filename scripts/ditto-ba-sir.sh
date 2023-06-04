cd .. && python3 ./ditto.py \
    --dataset ba-sir \
    --seed 123456789 \
    --data_dir input \
    --output output/ditto-ba-sir.pt \
    --device cuda:0 \
    --b_pI0 1e-6 \
    --b_pR0 1e-6 \
    --b_steps 500 \
    --b_lr 0.003 \
    --q_steps 500 \
    --q_lr 0.001 \
    --q_hid 16 \
    --q_gnn 3 \
    --q_mlp 2 \
    --q_samples 10 \
    --q_zlim 16 \
    --p_coef 1.0 \
    --t_samples 100 \
    --t_steps 10 \
    --t_keep 0.5
