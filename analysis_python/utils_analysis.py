import os
from tqdm.notebook import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as pth_transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from kornia.filters import box_blur
from PIL import Image

import utils
import vision_transformer as vits
from vision_transformer import DINOHead

"""
General functions
"""
def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def argmax_2d(x, random_choice=True):
    if random_choice:
        max_indices = np.argwhere(x == np.max(x))
        max_indices.shape
        num_max = len(max_indices)
        if num_max > 1:
            idx = np.random.choice(num_max, 1)
            return max_indices[idx][0]
        else:
            return max_indices[0]
    else:
        return np.asarray(np.unravel_index(np.argmax(x, axis=None), x.shape))

def padding(frame, patch_size=16):
    """
    e.g. 
    frame_pad = padding(np.random.rand(10, 24, 3), patch_size)
    frame_pad.shape -> (16, 32, 3)
    """
    h, w, _ = frame.shape
    hp = h % patch_size
    wp = w % patch_size
    if hp == 0:
        hpad = (0,0)
    else:
        hpad = (0, patch_size - hp)
    if wp == 0:
        wpad = (0,0)
    else:
        wpad = (0, patch_size - wp)
    return np.pad(frame, (hpad, wpad, (0, 0)))

"""
Functions for load vision transformer model
"""
def model_load(training_method, trial_num, depth, 
               patch_size=16, out_dim=65536, num_classes=1000, 
               models_dir="../trained_model_weights/",
               arch="vit_small", num_epochs=300):
    # getting data file path
    model_dir = models_dir + "{}/{:0=2}/{}layers".format(training_method, trial_num, depth)
    if num_epochs == 100:
        fname = "checkpoint0100.pth"
    elif num_epochs == 300:
        fname = "checkpoint.pth"
    weight_path = os.path.join(model_dir, fname)
    checkpoint = torch.load(weight_path, map_location="cpu")

    # loading models
    if training_method == "dino":
        model = vits.__dict__[arch](patch_size=patch_size, depth=depth, drop_path_rate=0.1)
        model.head = DINOHead(model.embed_dim, out_dim, use_bn=False, norm_last_layer=True) #, nlayers=3
        model_state_dict = {}
        for key, value in checkpoint["teacher"].items():
            if "projection_head" in key:
                model_state_dict[key.replace("projection_head", "mlp")] = value
            elif "prototypes" in key:
                model_state_dict[key.replace("prototypes", "last_layer")] = value
            else:
                model_state_dict[key] = value

        model_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}
        model_state_dict = {k.replace("backbone.", ""): v for k, v in model_state_dict.items()}
        model.load_state_dict(model_state_dict, strict=True)
    elif training_method == "supervised":
        # deit_small_patch16_224
        model = vits.__dict__[arch](patch_size=patch_size, depth=depth, drop_path_rate=0.1)
        model.head = nn.Linear(model.embed_dim, num_classes)
        model_state_dict = checkpoint["model"]
        model.load_state_dict(model_state_dict, strict=True)    
    else:
        assert False, "training_method must be dino or supervised"
        
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)
    return model, device

"""
Function and class for computation of head importance in self-attention
"""
class JSD(nn.Module):
    # https://discuss.pytorch.org/t/jensen-shannon-divergence/2626
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor):
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log()))

def get_head_importance(model, x, softmax_temp=0.1):
    # "dino": softmax_temp = 0.1, "supervised": softmax_temp = 1
    if model.training:
        model.eval()
    model_device = next(model.parameters()).device
    data_device = x.device
    if  model_device != data_device:
        x = x.to(model_device)

    # initialization
    depth = model.depth
    num_heads = model.num_heads
    head_importance = np.zeros((depth, num_heads))
    jsd = JSD() # get JSD

    # get actual output
    z_act = model.forward(x)
    y_act = model.head(z_act)
    out_act = F.softmax(y_act / softmax_temp, dim=-1)
    
    # get ablated outputs
    for depth_idx in range(depth):
        for head_idx in range(num_heads):
            z_abl = model.mask_forward(x, mask_depth_idx=depth_idx, mask_head_idx=head_idx)
            y_abl = model.head(z_abl)
            out_abl = F.softmax(y_abl / softmax_temp, dim=-1)
            head_importance[depth_idx, head_idx] = jsd(out_act, out_abl).detach().cpu().numpy()
    head_importance_normalize = head_importance / head_importance.sum(axis=1, keepdims=1)
    return head_importance, head_importance_normalize

"""
Function for computation of model gaze points
"""
class ImageDataset(Dataset):
    def __init__(self, image_path_list, transform=None):
        self.image_path_list = image_path_list
        self.transform = transform

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        img_path = self.image_path_list[idx]
        image = Image.open(img_path)
        if image.mode == "L":
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def get_gaze_pos_model(model, device, images, patch_size=16, blur_size=32):
    num_heads = model.num_heads
    with torch.inference_mode():
        attentions_list = model.get_fulllayers_selfattention(images.to(device))

    num_samples = len(images)
    w_featmap = images.shape[-2] // patch_size
    h_featmap = images.shape[-1] // patch_size
    
    gaze_pos_model = []
    for attentions in attentions_list:
        attentions = attentions[:, :, 0, 1:].reshape(num_samples, num_heads, w_featmap, h_featmap)
        attentions = nn.functional.interpolate(attentions, scale_factor=patch_size, mode="nearest")
        attentions_mean = torch.mean(attentions, axis=1, keepdims=True)
        attentions = torch.cat((attentions, attentions_mean), dim=1)
        attentions_blur = box_blur(attentions, blur_size).detach().cpu().numpy()
        
        gaze_pos = np.zeros((num_samples, 1, num_heads+1, 2))
        for b_idx in range(num_samples):
            for h_idx in range(num_heads+1):
                attn = attentions_blur[b_idx, h_idx]
                ey, ex = argmax_2d(attn)
                gaze_pos[b_idx, :, h_idx] = np.array([ex, ey])
        gaze_pos_model.append(gaze_pos)
    gaze_pos_model = np.concatenate(gaze_pos_model, 1)
    gaze_pos_model = gaze_pos_model.transpose(1,2,0,3)
    return gaze_pos_model

def get_gaze_pos_model_dataset(model, device, dataloader, patch_size=16, blur_size=32):
    num_heads = model.num_heads
    gaze_pos_model = []
    w_featmap, h_featmap = None, None
    for images in tqdm(dataloader):
        with torch.inference_mode():
            attentions_list = model.get_fulllayers_selfattention(images.to(device))
    
        num_samples = len(images)
        if w_featmap is None:
            w_featmap = images.shape[-2] // patch_size
            h_featmap = images.shape[-1] // patch_size
        
        gaze_pos_layers = []
        for attentions in attentions_list:
            attentions = attentions[:, :, 0, 1:].reshape(num_samples, num_heads, w_featmap, h_featmap)
            attentions = nn.functional.interpolate(attentions, scale_factor=patch_size, mode="nearest")
            attentions_mean = torch.mean(attentions, axis=1, keepdims=True)
            attentions = torch.cat((attentions, attentions_mean), dim=1)
            attentions_blur = box_blur(attentions, blur_size).detach().cpu().numpy()
            
            gaze_pos = np.zeros((num_samples, 1, num_heads+1, 2))
            for b_idx in range(num_samples):
                for h_idx in range(num_heads+1):
                    attn = attentions_blur[b_idx, h_idx]
                    ey, ex = argmax_2d(attn)
                    gaze_pos[b_idx, :, h_idx] = np.array([ex, ey])
            gaze_pos_layers.append(gaze_pos)
        gaze_pos_layers = np.concatenate(gaze_pos_layers, 1)
        gaze_pos_model.append(gaze_pos_layers)
    gaze_pos_model = np.concatenate(gaze_pos_model, 0)
    gaze_pos_model = gaze_pos_model.transpose(1,2,0,3)
    return gaze_pos_model

"""
Function for computation of viewing proportion
"""
def omitnan(x):
    return x[~np.isnan(x)]

# from https://stackoverflow.com/questions/40084931/taking-subarrays-from-numpy-array-with-given-stride-stepsize/
def strided_app(a, L, S ):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n,n))

def median_filter_1d(x, kernel_size=5):
    # x: 1d array
    # kernel_size should be odd number
    nan = np.ones(kernel_size-1) * np.nan
    x_nan = np.hstack([nan, x])
    return np.nanmedian(strided_app(x_nan, kernel_size, 1), axis=1)

def mean_filter_1d(x, kernel_size=5):
    # x: 1d array
    nan = np.ones(kernel_size-1) * np.nan
    x_nan = np.hstack([nan, x])
    return np.nanmean(strided_app(x_nan, kernel_size, 1), axis=1)

def get_viewing_prop(gaze_pos, obj_x_pos, obj_y_pos, sigma=30):
    num_frames = len(gaze_pos) # gaze_pos: (num_frames, 2)
    num_keypoints = len(obj_x_pos) # (num_keypoints, num_frames)
    gaze_weight = np.zeros((num_frames, num_keypoints)) # to output
    for frame_idx in range(num_frames):
        if not np.isnan(obj_x_pos[:, frame_idx]).all(): # key point is not nan
            x_key = omitnan(obj_x_pos[:, frame_idx])
            y_key = omitnan(obj_y_pos[:, frame_idx])
            num_not_nan = len(x_key)
            
            x_eye, y_eye = gaze_pos[frame_idx]
            d = np.exp(- ((x_key - x_eye)**2 + (y_key - y_eye)**2) / (2*(sigma**2)))
            if np.sum(d) > 1:
                gaze_weight[frame_idx, :num_not_nan] = d / np.sum(d) # normalization & save
            elif np.sum(d) <= 1 and np.sum(d) > 0:
                gaze_weight[frame_idx, :num_not_nan] = d
    return gaze_weight
    
"""
def get_viewing_prop(model_eye_data, xx_all, yy_all, num_keypoints=18, sigma=30):
    model_eye_pos_all = model_eye_data['gaze_point']
    num_models, depth, num_frames, _ = model_eye_pos_all.shape
    print(num_models, depth, num_frames, num_keypoints)
    gaze_weight_model = np.zeros((num_models, depth, num_frames, num_keypoints)) # to output
    #filtered_model_eye_pos = np.apply_along_axis(mean_filter_1d, 3, model_eye_pos_all[:,:,:,:,blur_idx], kernel_size=6)
    for frame_idx in tqdm(range(num_frames)):
        if not np.isnan(xx_all[:, frame_idx]).all(): # key point is not nan
            x_key = omitnan(xx_all[:, frame_idx])
            y_key = omitnan(yy_all[:, frame_idx])
            num_not_nan = len(x_key)
            for trial_idx in range(num_models):
                for depth_idx in range(depth):
                    x_eye, y_eye = model_eye_pos_all[trial_idx, depth_idx, frame_idx]
                    d = np.exp(- ((x_key - x_eye)**2 + (y_key - y_eye)**2) / (2*(sigma**2)))
                    if np.sum(d) > 1:
                        gaze_weight_model[trial_idx, depth_idx, frame_idx, :num_not_nan] = d / np.sum(d) # normalization & save
                    elif np.sum(d) <= 1 and np.sum(d) > 0:
                        gaze_weight_model[trial_idx, depth_idx, frame_idx, :num_not_nan] = d
    return gaze_weight_model
"""