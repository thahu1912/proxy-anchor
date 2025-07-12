python train.py --gpu-id 0 \
                --loss Proxy_Anchor \
                --model bn_inception \
                --embedding-size 512 \
                --batch-size 180 \
                --lr 1e-4 \
                --dataset cars \
                --warm 1 \
                --bn-freeze 1 \
                --lr-decay-step 20

python train.py --gpu-id 0 \
                --loss Uncertainty_Aware_Proxy_Anchor \
                --model bn_inception \
                --embedding-size 512 \
                --batch-size 180 \
                --lr 1e-4 \
                --dataset cars \
                --warm 1 \
                --bn-freeze 1 \
                --lr-decay-step 20 \
                --variance-weight 0.05 \
                --hyper-weight 0.3

####Proxy Anchor-64####
####Proxy Anchor-BN_Inception####
python train.py --gpu-id 0 \
                --loss Proxy_Anchor \
                --model bn_inception \
                --embedding-size 64 \
                --batch-size 180 \
                --lr 1e-4 \
                --dataset cub \
                --warm 1 \
                --bn-freeze 1 \
                --lr-decay-step 10


python train.py --gpu-id 0 \
                --loss Uncertainty_Aware_Proxy_Anchor \
                --model bn_inception \
                --embedding-size 64 \
                --batch-size 180 \
                --lr 1e-4 \
                --dataset cars \
                --warm 1 \
                --bn-freeze 1 \
                --lr-decay-step 20 \
                --variance-weight 0.05 \
                --hyper-weight 0.3

