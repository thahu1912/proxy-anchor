###################CUB####################

####################BN_Inception####################
python train.py --gpu-id 0 \
                --loss Uncertainty_Aware_Proxy_Anchor \
                --model bn_inception \
                --embedding-size 512 \
                --batch-size 180 \
                --lr 1e-4 \
                --dataset cub \
                --warm 1 \
                --bn-freeze 1 \
                --lr-decay-step 10
                --variance-weight 0.1 \
                --hyper-weight 0.2 \



python train.py --gpu-id 0 \
                --loss Proxy_Anchor \
                --model bn_inception \
                --embedding-size 512 \
                --batch-size 180 \
                --lr 1e-4 \
                --dataset cub \
                --warm 1 \
                --bn-freeze 1 \
                --lr-decay-step 10




###################GoogleNet####################
python train.py --gpu-id 0 \
                --loss Uncertainty_Aware_Proxy_Anchor \
                --model googlenet \
                --embedding-size 512 \
                --batch-size 180 \
                --lr 1e-4 \
                --dataset cub \
                --warm 1 \
                --bn-freeze 1 \
                --lr-decay-step 10
                --variance-weight 0.1 \
                --hyper-weight 0.2 \



python train.py --gpu-id 0 \
                --loss Proxy_Anchor \
                --model googlenet \
                --embedding-size 512 \
                --batch-size 180 \
                --lr 1e-4 \
                --dataset cub \
                --warm 1 \
                --bn-freeze 1 \
                --lr-decay-step 10

###################Cars####################
####################BN_Inception####################
python train.py --gpu-id 0 \
                --loss Uncertainty_Aware_Proxy_Anchor \
                --model bn_inception \
                --embedding-size 512 \
                --batch-size 180 \
                --lr 1e-4 \
                --dataset cars \
                --warm 1 \
                --bn-freeze 1 \
                --lr-decay-step 10
                --variance-weight 0.1 \
                --hyper-weight 0.2 \



python train.py --gpu-id 0 \
                --loss Proxy_Anchor \
                --model bn_inception \
                --embedding-size 512 \
                --batch-size 180 \
                --lr 1e-4 \
                --dataset cars \
                --warm 1 \
                --bn-freeze 1 \
                --lr-decay-step 10


###################GoogleNet####################
python train.py --gpu-id 0 \
                --loss Uncertainty_Aware_Proxy_Anchor \
                --model googlenet \
                --embedding-size 512 \
                --batch-size 180 \
                --lr 1e-4 \
                --dataset cars \
                --warm 1 \
                --bn-freeze 1 \
                --lr-decay-step 10
                --variance-weight 0.1 \
                --hyper-weight 0.2 \



python train.py --gpu-id 0 \
                --loss Proxy_Anchor \
                --model googlenet \
                --embedding-size 512 \
                --batch-size 180 \
                --lr 1e-4 \
                --dataset cars \
                --warm 1 \
                --bn-freeze 1 \
                --lr-decay-step 10


###################SOP####################
####################BN_Inception####################
python train.py --gpu-id 0 \
                --loss Uncertainty_Aware_Proxy_Anchor \
                --model bn_inception \
                --embedding-size 512 \
                --batch-size 180 \
                --lr 1e-4 \
                --dataset SOP \
                --warm 1 \
                --bn-freeze 1 \
                --lr-decay-step 10
                --variance-weight 0.1 \
                --hyper-weight 0.2 \



python train.py --gpu-id 0 \
                --loss Proxy_Anchor \
                --model bn_inception \
                --embedding-size 512 \
                --batch-size 180 \
                --lr 1e-4 \
                --dataset SOP \
                --warm 1 \
                --bn-freeze 1 \
                --lr-decay-step 10



###################GoogleNet####################
python train.py --gpu-id 0 \
                --loss Uncertainty_Aware_Proxy_Anchor \
                --model googlenet \
                --embedding-size 512 \
                --batch-size 180 \
                --lr 1e-4 \
                --dataset SOP \
                --warm 1 \
                --bn-freeze 1 \
                --lr-decay-step 10
                --variance-weight 0.1 \
                --hyper-weight 0.2 \



python train.py --gpu-id 0 \
                --loss Proxy_Anchor \
                --model googlenet \
                --embedding-size 512 \
                --batch-size 180 \
                --lr 1e-4 \
                --dataset SOP \
                --warm 1 \
                --bn-freeze 1 \
                --lr-decay-step 10

