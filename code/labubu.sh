####VonMisesFisher_Proxy_Anchor-512####

####################BN_Inception####################
# Recommended grid search for SOP dataset (embedding size 512)
for kappa in 0.5 1.0 2.0; do
  for temp in 0.05 0.1 0.2; do
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
                    --concentration-init $kappa \
                    --lr-decay-gamma 0.25 \
                    --temperature $temp \
    echo "Finished SOP 512: kappa=$kappa, temp=$temp"
  done
done

####VonMisesFisher_Proxy_Anchor-64####

####################BN_Inception####################
# Recommended grid search for SOP dataset (embedding size 64)
for kappa in 0.5 1.0 2.0; do
  for temp in 0.05 0.1 0.2; do
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
                    --concentration-init $kappa \
                    --lr-decay-gamma 0.25 \
                    --temperature $temp \
    echo "Finished SOP 64: kappa=$kappa, temp=$temp"
  done
done












