import os.path
from pdb import set_trace
from data.base_dataset import BaseDataset, get_params, get_transform
from data.np_image_folder import make_dataset
import numpy as np
from PIL import Image

"""A custom datasat to load numpy iamges. Will retain most of the 
functionality of the original dataset class of which is not dependent on
the PIL image class. It is also written specifically to wotk with images
where no label map is given (e.g. with the no_instance set to true) and
label_nc = 0. The folder structure is also changed from dataroot/phase+fold
to dataroot/phase/fold where fold is either A or B.
"""
class NumpyAlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.validate_options()
        self.root = opt.dataroot    

        ### input A (label maps)
        
        self.dir_A = os.path.join(opt.dataroot, *[opt.phase, 'A'])
        self.A_paths = sorted(make_dataset(self.dir_A))

        ### input B (real images)
        if opt.isTrain or opt.use_encoded_image:
            self.dir_B = os.path.join(opt.dataroot, *[opt.phase, 'B'])
            self.B_paths = sorted(make_dataset(self.dir_B))

        self.dataset_size = len(self.A_paths) 
      
    def __getitem__(self, index):        
        ### input A (label maps)
        A_path = self.A_paths[index]              
        A = np.load(A_path)
        params = get_params(self.opt, A.shape[:2])
        if self.opt.label_nc == 0: # If no labels are used
            transform_A = get_transform(self.opt, params, normalize=False)
            A_tensor = transform_A(A)
        else: # With labels we use a nearest neighbour interpolator for resizing.
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            A_tensor = transform_A(A)

        B_tensor = inst_tensor = feat_tensor = 0
        ### input B (real images)
        if self.opt.isTrain or self.opt.use_encoded_image:
            B_path = self.B_paths[index]   
            B = np.load(B_path)
            B = np.stack((B,)*3, axis=-1) if B.ndim == 2 else B
            transform_B = get_transform(self.opt, params, normalize=False)      

            B_tensor = transform_B(B)

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 
                      'feat': feat_tensor, 'path': A_path}

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'NpArrayAlignedDataset'

    def validate_options(self):
        expected_values = {
            'label_nc' : 0,
            'no_instance': True,
            'image_format' : 'npy',
            'load_features' : False
        }
        if self.opt.phase == 'test':
            expected_values['engine'] = None
            expected_values['onnx'] = None

        for key in expected_values.keys():
            assert expected_values[key] == getattr(self.opt, key), \
                f'Expected the value {expected_values[key]} for {key} ' \
                f'got {getattr(self.opt, key)}. '\
                'See data/np_aligned_dataset.py for a summary of '\
                'the exepected values.'