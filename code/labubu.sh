####VonMisesFisher_Proxy_Anchor-512####

####################BN_Inception####################
python train.py --gpu-id 0 \
                --loss VonMisesFisher_Proxy_Anchor \
                --model bn_inception \
                --embedding-size 512 \
                --batch-size 180 \
                --lr 1e-5 \
                --dataset cub \
                --warm 1 \
                --bn-freeze 1 \
                --lr-decay-step 10 \
                --concentration-init 1.0 \
                --temperature 0.1 \
                
                
###################Cars####################
python train.py --gpu-id 0 \
                --loss VonMisesFisher_Proxy_Anchor \
                --model bn_inception \
                --embedding-size 512 \
                --batch-size 180 \
                --lr 1e-4 \
                --dataset cars \
                --warm 1 \
                --bn-freeze 1 \
                --lr-decay-step 20 \
                --concentration-init 1.0 \
                --temperature 0.1 \
                


###################SOP####################
####################BN_Inception####################
python train.py --gpu-id 0 \
                --loss VonMisesFisher_Proxy_Anchor \
                --model bn_inception \
                --embedding-size 512 \
                --batch-size 180 \
                --lr 6e-4 \
                --dataset SOP \
                --warm 1 \
                --bn-freeze 1 \
                --lr-decay-step 20 \
                --concentration-init 0.8 \
                --lr-decay-gamma 0.25\
                --temperature 0.5 \
                


####VonMisesFisher_Proxy_Anchor-64####

####################BN_Inception####################
python train.py --gpu-id 0 \
                --loss VonMisesFisher_Proxy_Anchor \
                --model bn_inception \
                --embedding-size 64 \
                --batch-size 180 \
                --lr 1e-4 \
                --dataset cub \
                --warm 1 \
                --bn-freeze 1 \
                --lr-decay-step 10 \
                --concentration-init 1 \
                --temperature 0.5 \
                
                
###################Cars####################
python train.py --gpu-id 0 \
                --loss VonMisesFisher_Proxy_Anchor \
                --model bn_inception \
                --embedding-size 64 \
                --batch-size 180 \
                --lr 1e-4 \
                --dataset cars \
                --warm 1 \
                --bn-freeze 1 \
                --lr-decay-step 20 \
                --concentration-init 1 \
                --temperature 0.5 \
                


###################SOP####################
####################BN_Inception####################
python train.py --gpu-id 0 \
                --loss VonMisesFisher_Proxy_Anchor \
                --model bn_inception \
                --embedding-size 64 \
                --batch-size 180 \
                --lr 6e-4 \
                --dataset SOP \
                --warm 1 \
                --bn-freeze 1 \
                --lr-decay-step 20 \
                --concentration-init 1 \
                --lr-decay-gamma 0.25\
                --temperature 0.5 \
                












