cd .. && python3 ./gin.py \
    --dataset er-si \
    --seed 123456789 \
    --data_dir input \
    --device cuda:0 \
    --output output/gin-er-si.pt \
    --b_pI0 0.001 \
    --b_pR0 0.001 \
    --b_pImax 0.999 \
    --b_steps 500 \
    --b_lr 0.003 \
    --lr 0.001 \
    --epochs 1000 \
    --batch_size 16 \
    --units 64 \
    --layers 3 \
    --dropout 0.1