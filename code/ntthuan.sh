####Uncertainty_Aware_Proxy_Anchor-512####

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
                --lr-decay-step 10 \
                --variance-weight 0.2 \
                --hyper-weight 0.5 



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
                --variance-weight 0.2 \
                --hyper-weight 0.5 


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
                --variance-weight 0.2 \
                --hyper-weight 0.5 


####Uncertainty_Aware_Proxy_Anchor-64####

####################BN_Inception####################
python train.py --gpu-id 0 \
                --loss Uncertainty_Aware_Proxy_Anchor \
                --model bn_inception \
                --embedding-size 64 \
                --batch-size 180 \
                --lr 1e-4 \
                --dataset cub \
                --warm 1 \
                --bn-freeze 1 \
                --lr-decay-step 10 \
                --variance-weight 0.2 \
                --hyper-weight 0.5 \



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
                --variance-weight 0.2 \
                --hyper-weight 0.5 


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
                --variance-weight 0.2 \
                --hyper-weight 0.5 





