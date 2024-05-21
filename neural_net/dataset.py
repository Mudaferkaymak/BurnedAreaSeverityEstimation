import os
import re
import numpy as np
import pandas as pd

from .scanner import Scanner
from .image_processor import ImageProcessor, ProductProcessor
from torch.utils.data.dataset import Dataset

from collections import OrderedDict

class SatelliteDataset(Dataset):
    def __init__(self, folder, mask_intervals: list, mask_one_hot: bool, height: int, width: int, product_list: list, mode, filter_validity_mask, transform=None, process_dict: dict=None, activation_date_csv: str=None, folder_list: list=None, ignore_list: list=None, mask_filtering: bool=False, only_burnt: bool=False, mask_postfix: str='mask'):
        self.folder = folder
        self.mask_intervals = mask_intervals
        self.mask_one_hot = mask_one_hot
        self.height = height
        self.width = width
        self.product_list = product_list
        self.mode = mode
        self.filter_validity_mask = filter_validity_mask
        self.mask_filtering = mask_filtering
        self.only_burnt = only_burnt
        self.mask_postfix = mask_postfix
        self.apply_mask_discretization = True
        if self.mask_postfix != 'mask':
            self.apply_mask_discretization = False

        if isinstance(filter_validity_mask, bool):
            self.filter_validity_flag = True
        elif isinstance(filter_validity_mask, list) or isinstance(filter_validity_mask, set):
            self.filter_validity_flag = True
        elif filter_validity_mask is None:
            self.filter_validity_flag = False
        else:
            raise ValueError('Invalid value for filter_validity_mask %s' % str(filter_validity_mask))

        self.transform = transform
        self.process_dict = process_dict
        self.activation_date_csv = activation_date_csv

        self.folder_list = folder_list
        self.ignore_list = ignore_list

        if self.folder_list is None:
            self.folder_list = []
            for dirname in os.listdir(self.folder):
                if self.ignore_list is not None and dirname in self.ignore_list:
                    continue

                self.folder_list.append(dirname)

        self.scanner = Scanner(folder, product_list, activation_date_csv, mask_intervals=mask_intervals, mask_one_hot=mask_one_hot, ignore_list=self.ignore_list, valid_list=self.folder_list)
        self.processor = ImageProcessor(height, width)

        regexp = r'^(((?!pre|post).)+)(_(pre|post))?$'
        regexp = re.compile(regexp, re.IGNORECASE)

        self.images = []
        self.masks = []
        for idx, dirname in enumerate(self.folder_list):
            image, product = self.scanner.get_all(dirname, self.product_list, mode=mode, retrieve_mask=True, mask_postfix=self.mask_postfix, apply_mask_discretization=self.apply_mask_discretization)
            brnt_mask = self.scanner.get_mask(dirname)
            assert product[-1] == 'mask'

            if self.only_burnt:
                tmp_mask = brnt_mask
                if self.mask_one_hot:
                    tmp_mask = tmp_mask.argmax(axis=-1, keepdims=True)
                if not ((tmp_mask >= 1).any()):
                    continue
                
            if self.filter_validity_flag:
                for img, prod in zip(image, product):
                    search_res = regexp.search(prod)
                    if not search_res:
                        raise ValueError('Invalid product name encountered %s' % prod)
                    base_prod = search_res.group(1)
                    if prod == 'mask':
                        continue
                    if (isinstance(self.filter_validity_mask, bool) and self.filter_validity_mask) or ((isinstance(self.filter_validity_mask, list) or isinstance(self.filter_validity_mask, set)) and base_prod in self.filter_validity_mask):
                        bool_mask = img[:, :, -1] == 0
                        img[:, :, :-1][bool_mask] = 0

            channel_counter = [x.shape[-1] for x in image]
            image = self.processor.upscale(image, product, dirname, concatenate=False)
            brnt_mask = self.processor.upscale([brnt_mask], ['mask'], dirname, concatenate=False)
            if process_dict is not None:
                image, product = self.processor.process(image, product, process_dict, return_ndarray=True, channel_counter=channel_counter)

            if self.mask_filtering:
                tmp_mask = brnt_mask[0]
                nonburned_mask = tmp_mask == 0 if not self.mask_one_hot else tmp_mask[..., 0] == 1
                if len(nonburned_mask.shape) == 3:
                    nonburned_mask = nonburned_mask.squeeze(axis=-1)
                if isinstance(image, np.ndarray):
                    image[..., :-1][nonburned_mask] = 0
                elif isinstance(image, list):
                    for idx, tmp_img in enumerate(image):
                        if idx != (len(image) - 1):
                            image[idx][nonburned_mask] = 0
                else:
                    raise ValueError('Invalid image type %s' % str(type(image)))

            image, count = self.processor.cut(image, product, return_ndarray=True, apply_mask_round=self.apply_mask_discretization)
            image, mask = image[..., :-1], image[..., -1:]

            brnt_mask, _ = self.processor.cut(brnt_mask, ['mask'], return_ndarray=True, apply_mask_round=True)

            if self.only_burnt:
                tmp_mask = brnt_mask
                if self.mask_one_hot:
                    tmp_mask = tmp_mask.argmax(axis=-1, keepdims=True)
                valid_cut = (tmp_mask > 0).any(axis=(1, 2))
                valid_cut = np.arange(valid_cut.shape[0])[valid_cut.flatten()]
                image = image[valid_cut, ...]
                mask = mask[valid_cut, ...]

            self.images.append(image)
            self.masks.append(mask)

        assert len(self.images) == len(self.masks)

        self.images = np.concatenate(self.images, axis=0)
        self.masks = np.concatenate(self.masks, axis=0)

        assert self.images.shape[0] == self.masks.shape[0]
        return

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        result = {}
        result['image'] = self.images[idx]
        result['mask'] = self.masks[idx]

        if self.transform is not None:
            result = self.transform(result)
        return result
