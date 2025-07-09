


# Proxy_Anchor
python /code/train.py --gpu-id 0 \
                --loss Proxy_Anchor \
                --model bn_inception \
                --embedding-size 512 \
                --batch-size 180 \
                --lr 1e-4 \
                --dataset cub \
                --warm 1 \
                --bn-freeze 1 \
                --lr-decay-step 10

python /code/train.py --gpu-id 0 \
                --loss Proxy_Anchor \
                --model bn_inception \
                --embedding-size 512 \
                --batch-size 180 \
                --lr 1e-4 \
                --dataset cars \
                --warm 1 \
                --bn-freeze 1 \
                --lr-decay-step 10

python /code/train.py --gpu-id 0 \
                --loss Proxy_Anchor \
                --model googlenet \
                --embedding-size 512 \
                --batch-size 180 \
                --lr 1e-4 \
                --dataset SOP \
                --warm 1 \
                --bn-freeze 1 \
                --lr-decay-step 10


# Uncertainty_Aware_Proxy_Anchor
python /code/train.py --gpu-id 0 \
                --loss Uncertainty_Aware_Proxy_Anchor \
                --model bn_inception \
                --embedding-size 512 \
                --batch-size 180 \
                --lr 1e-4 \
                --dataset cub \
                --warm 1 \
                --bn-freeze 1 \
                --lr-decay-step 10

python /code/train.py --gpu-id 0 \
                --loss Uncertainty_Aware_Proxy_Anchor \
                --model resnet50 \
                --embedding-size 512 \
                --batch-size 180 \
                --lr 1e-4 \
                --dataset cars \
                --warm 1 \
                --bn-freeze 1 \
                --lr-decay-step 10

python /code/train.py --gpu-id 0 \
                --loss Uncertainty_Aware_Proxy_Anchor \
                --model googlenet \
                --embedding-size 512 \
                --batch-size 180 \
                --lr 1e-4 \
                --dataset SOP \
                --warm 1 \
                --bn-freeze 1 \
                --lr-decay-step 10