
python train.py --gpu-id 0 \
                --loss Proxy_Anchor \
                --model bn_inception \
                --embedding-size 512 \
                --batch-size 180 \
                --lr 6e-4 \
                --dataset SOP \
                --warm 1 \
                --bn-freeze 0 \
                --lr-decay-step 20 \
                --lr-decay-gamma 0.25


python train.py --gpu-id 0 \
                --loss Proxy_Anchor \
                --model bn_inception \
                --embedding-size 512 \
                --batch-size 180 \
                --lr 6e-4 \
                --dataset SOP \
                --warm 1 \
                --bn-freeze 0 \
                --lr-decay-step 20 \
                --lr-decay-gamma 0.25


python train.py --gpu-id 0 \
                --loss Uncertainty_Aware_Proxy_Anchor \
                --model bn_inception \
                --embedding-size 512 \
                --batch-size 180 \
                --lr 6e-4 \
                --dataset SOP \
                --warm 1 \
                --bn-freeze 1 \
                --lr-decay-step 20 \
                --lr-decay-gamma 0.25 \
                --variance-weight 0.15 \
                --hyper-weight 0.7


python train.py --gpu-id 0 \
                --loss Uncertainty_Aware_Proxy_Anchor \
                --model bn_inception \
                --embedding-size 64 \
                --batch-size 180 \
                --lr 6e-4 \
                --dataset SOP \
                --warm 1 \
                --bn-freeze 1 \
                --lr-decay-step 20 \
                --lr-decay-gamma 0.25 \
                --variance-weight 0.15 \
                --hyper-weight 0.7




