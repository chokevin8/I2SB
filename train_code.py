
# conditional model training with segmentation map as conditioning (continued training at iter 2184)
python train.py --ckpt "test-run-seg-map-cond" --name "test-run-seg-map-cond" --cond-x1 --num-itr 36400 --lr-step 364 --corrupt "mixture" --save-pt-every 728 --val-every 728 --image-size 256 --dataset-dir /run/user/1000/gvfs/smb-share:server=shelter.local,share=kyu/IHC2HE/Balanced_Aligned/dataset_v1_256x256/ --batch-size 128 --microbatch 1 --beta-max 0.3 --log-dir /home/labuser1/PycharmProjects/I2SB/logs --log-writer "wandb" --wandb-api-key "056cc8f5fd5428b2d91107a96c0da5f5ae1c3476" --wandb-user "chokevin8"



# # try running unconditional above but with NS2HE dataset now, same hyperparameters as test-run-4 above, modified for # of datasets: #1,014,656 imgs, 128 batch size, 7927 iters = 1 epoch
# python train.py --name "NS2HE-test-run-1" --num-itr 792700 --lr-step 7927 --corrupt "mixture" --save-pt-every 7927 --val-every 7927 --image-size 256 --dataset-dir /run/user/1000/gvfs/smb-share:server=shelter.local,share=kyu/unstain2stain/tiles/registered_tiles/I2SB --batch-size 128 --microbatch 1 --beta-max 0.3 --log-dir /home/labuser1/PycharmProjects/I2SB/logs --log-writer "wandb" --wandb-api-key "056cc8f5fd5428b2d91107a96c0da5f5ae1c3476" --wandb-user "chokevin8"
# # resume:
# python train.py --ckpt "NS2HE-test-run-1" --name "NS2HE-test-run-1" --num-itr 792700 --lr-step 7927 --corrupt "mixture" --save-pt-every 7927 --val-every 7927 --image-size 256 --dataset-dir /run/user/1000/gvfs/smb-share:server=shelter.local,share=kyu/unstain2stain/tiles/registered_tiles/I2SB --batch-size 128 --microbatch 1 --beta-max 0.3 --log-dir /home/labuser1/PycharmProjects/I2SB/logs --log-writer "wandb" --wandb-api-key "056cc8f5fd5428b2d91107a96c0da5f5ae1c3476" --wandb-user "chokevin8"


# 23456 images of 256 x 256, batch size = 128, so 23456/128 = 183 iterations per epoch. And then train for 100 epoch. 183 * 100 = 18300 total # of iterations
# lr-step = num-itr/100
#HE2IHC cond I2sb training:
python train.py --name "HE2IHC_cond" --dataset-dir /home/labuser1/Desktop/256x256 --num-itr 18300 --lr-step 183 --save-pt-every 3660 --val-every 3660 --cond-x1 --corrupt "mixture" --image-size 256  --batch-size 128 --microbatch 1 --beta-max 0.3 --log-dir /home/labuser1/PycharmProjects/I2SB/logs --log-writer "wandb" --wandb-api-key "056cc8f5fd5428b2d91107a96c0da5f5ae1c3476" --wandb-user "chokevin8"
