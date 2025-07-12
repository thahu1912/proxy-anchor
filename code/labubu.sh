####Uncertainty_Aware_Proxy_Anchor_V2-512####

####################BN_Inception####################
python train.py --gpu-id 0 \
                --loss Uncertainty_Aware_Proxy_Anchor_V2 \
                --model bn_inception \
                --embedding-size 512 \
                --batch-size 180 \
                --lr 1e-4 \
                --dataset cub \
                --warm 1 \
                --bn-freeze 1 \
                --lr-decay-step 10 \
                --variance-weight 0.1 \
                --hyper-weight 0.2 \
                
                
###################Cars####################
python train.py --gpu-id 0 \
                --loss Uncertainty_Aware_Proxy_Anchor_V2 \
                --model bn_inception \
                --embedding-size 512 \
                --batch-size 180 \
                --lr 1e-4 \
                --dataset cars \
                --warm 1 \
                --bn-freeze 1 \
                --lr-decay-step 20 \
                --lr-decay-gamma 0.25 \
                --variance-weight 0.1 \
                --hyper-weight 0.2 \
                


###################SOP####################
####################BN_Inception####################
python train.py --gpu-id 0 \
                --loss Uncertainty_Aware_Proxy_Anchor_V2 \
                --model bn_inception \
                --embedding-size 512 \
                --batch-size 180 \
                --lr 6e-4 \
                --dataset SOP \
                --warm 1 \
                --bn-freeze 1 \
                --lr-decay-step 20 \
                --lr-decay-gamma 0.25 \
                --variance-weight 0.1 \
                --hyper-weight 0.2 \
                


####Uncertainty_Aware_Proxy_Anchor_V2-64####

####################BN_Inception####################
python train.py --gpu-id 0 \
                --loss Uncertainty_Aware_Proxy_Anchor_V2 \
                --model bn_inception \
                --embedding-size 64 \
                --batch-size 180 \
                --lr 1e-4 \
                --dataset cub \
                --warm 1 \
                --bn-freeze 1 \
                --lr-decay-step 10 \
                --variance-weight 0.1 \
                --hyper-weight 0.2 \
                
                
###################Cars####################
python train.py --gpu-id 0 \
                --loss Uncertainty_Aware_Proxy_Anchor_V2 \
                --model bn_inception \
                --embedding-size 64 \
                --batch-size 180 \
                --lr 1e-4 \
                --dataset cars \
                --warm 1 \
                --bn-freeze 1 \
                --lr-decay-step 20 \
                --variance-weight 0.1 \
                --hyper-weight 0.2 \
                


###################SOP####################
####################BN_Inception####################
python train.py --gpu-id 0 \
                --loss Uncertainty_Aware_Proxy_Anchor_V2 \
                --model bn_inception \
                --embedding-size 64 \
                --batch-size 180 \
                --lr 6e-4 \
                --dataset SOP \
                --warm 1 \
                --bn-freeze 1 \
                --lr-decay-step 20 \
                --lr-decay-gamma 0.25\
                --variance-weight 0.1 \
                --hyper-weight 0.2 \
                












