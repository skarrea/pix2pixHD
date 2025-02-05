from numpy.core.records import array
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as ft
import numpy as np
import random

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.resize_or_crop == 'resize_and_crop':
        new_h = new_w = opt.loadSize            
    elif opt.resize_or_crop == 'scale_width_and_crop':
        new_w = opt.loadSize
        new_h = opt.loadSize * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.fineSize))
    y = random.randint(0, np.maximum(0, new_h - opt.fineSize))
    
    flip = random.random() > 0.5
    return {'crop_pos': (x, y), 'flip': flip}

def get_transform(opt, params, method=Image.BICUBIC, normalize=True):
    transform_list = []

    transform_list += [transforms.Lambda(lambda img: transforms.ToTensor()(np.array(img)))] 
    # Converted to numpy array first to supress exessive error messages.
    
    if 'resize' in opt.resize_or_crop:
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Scale(osize, method))   
    elif 'scale_width' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.loadSize, method)))
        
    if 'crop' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.fineSize)))

    if opt.resize_or_crop == 'none':
        base = float(2 ** opt.n_downsample_global)
        if opt.netG == 'local':
            base *= (2 ** opt.n_local_enhancers)
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))


    if normalize:
        transform_list.append(transforms.Lambda(lambda img: __normalize(img)))

    return transforms.Compose(transform_list)

def __normalize(img):   
    return ft.normalize(img, img.shape[0]*(.5,), img.shape[0]*(.5,))

def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size        
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return transforms.Resize((w, h), method)

def __scale_width(img, target_width, method=Image.BICUBIC):
    oh, ow = img.shape[1:]
    if (ow == target_width):
        return img    
    w = target_width
    h = int(target_width * oh/ow)   
    return ft.resize(img, (h, w), method)

def __crop(img, pos, size):
    ow, oh = img.shape[1:]
    x1, y1 = pos
    tw = th = size
    
    if (ow > tw or oh > th):    
        img = ft.crop(img, y1, x1, th, tw)
        return ft.pad(img, [0, 0,  th-img.shape[2], tw - img.shape[1]])
    return img

def __flip(img, flip):
    if flip:
        return ft.hflip(img)
    return img
    
def normalize():    
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))