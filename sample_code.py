######################################### HE2IHC code

#HE2IHC sampling
python sample.py --nfe 62 --corrupt "mixture" --cond-x1 --save-every 50 --ckpt "HE2IHC_cond" --n-gpu-per-node 1 --dataset-dir /home/labuser1/Desktop/256x256 --batch-size 1
