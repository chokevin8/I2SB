import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
import os
from natsort import natsorted

# after running sample.py code for IHC2HE below:

#postprocess code:
def post_process_ddpm_sampling(pt_path, run_name, iter_name, pred_save_path, validation_HE_path):
    '''
    Input:
    pt_path: Path to .pt file, which is output of sample.py from I2SB, which contains prediction tensors.
    pred_save_path, run_name, iter_name: os.path.join(pred_save_path,run_name,iter_name) = Path to save inferred images to.
    validation_HE_path: Path that contains the validation HE images (GT image). Used to extract names. Make sure it has same order as the path for validation IHC images.
    '''
    images = torch.load(pt_path) #images["label_arr"] holds the GT image, but this is already stored in the val/HE folder.
    arrs = images["arr"]  #the order is the same as how the file is sorted in val/HE folder, so we find the respective image names from there.
    total_num_imgs = arrs.size()[0]
    all_val_image_names = [x.replace(".png","_inferred.png") for x in natsorted(os.listdir(validation_HE_path)) if x.endswith(".png")]
    pred_save_path = os.path.join(pred_save_path, run_name)
    if not os.path.exists(pred_save_path):
        os.makedirs(pred_save_path)
    pred_save_path = os.path.join(pred_save_path,iter_name)
    if not os.path.exists(pred_save_path):
        os.makedirs(pred_save_path)
    for idx in tqdm(range(0,total_num_imgs),colour= "red", desc = "Number of images"):
        pred_image = arrs[idx].permute(1,2,0).numpy()
        pred_image = (pred_image + 1) / 2.0
        pred_image = (pred_image * 255).astype(np.uint8)
        pred_image_save_path = os.path.join(pred_save_path,all_val_image_names[idx])
        Image.fromarray(pred_image).save(pred_image_save_path)
    print("Images all successfully saved!")

post_process_ddpm_sampling(pt_path = "/home/labuser1/PycharmProjects/I2SB/results/HE2IHC_cond/samples_nfe62/recon.pt",
                           run_name = "test-run",
                           iter_name = "iter_latest",
                           pred_save_path = "/home/labuser1/Desktop/256x256",
                           validation_HE_path = "/home/labuser1/Desktop/256x256/val/IHC")