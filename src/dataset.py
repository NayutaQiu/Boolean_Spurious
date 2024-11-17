import os
import numpy as np
import pandas as pd
import torch
import sys
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.utils.data.sampler import WeightedRandomSampler
import random
from transformers import BertTokenizer
import wilds
from utils import *
from tqdm import tqdm
from dataset_utils import *
import math
import torchvision
from torchvision.datasets.utils import (
    extract_archive,
)
from sklearn.utils.class_weight import compute_class_weight
import wget
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

class SpuriousBooleanSampling(IterableDataset):
    def __init__(self, core_len, spurious_len, core_func, spurious_func, c, sample_num, 
                 batch_size=64, sampling_method="pure", bypass_bias_check=False, device="cpu") -> None:
        # always output {+1,0}
        super().__init__()
        self.device = device
        BIAS_TESTING_SIZE = 5000
        assert sampling_method in ["pure", "on_request", "buffer", "auto"]
        if sampling_method == "auto":
            if spurious_len <= 15:
                sampling_method = "pure"
            else:
                sampling_method = "on_request"
        self.sampling_method = sampling_method
        test_x = generate_random_x(spurious_len, sample_num=BIAS_TESTING_SIZE, F_2=False).to(self.device)  
        test_y = func_batch_forward(spurious_func, test_x, device=self.device)
        self.pos_ratio = ((test_y+1)/2).mean()
        print(f"Spurious function has bias ratio: {self.pos_ratio}")
        if not bypass_bias_check:
            if abs(2*self.pos_ratio - 1) >= 0.4:
                raise Exception("The function is heavily biased and may affect speed")
        if sampling_method == "pure":
            assert core_len <= 20 and spurious_len <= 20
        # First compute all spurious y.
            spurious_x = generate_random_x(spurious_len, F_2=False).to(self.device) 
            spurious_y = func_batch_forward(spurious_func, spurious_x, device=self.device).to(self.device) 
            self.spurious_pos_x = spurious_x[spurious_y==1]
            self.spurious_neg_x = spurious_x[spurious_y==-1]
        if sampling_method == "buffer":
            raise NotImplementedError("buffer method not implemented yet")
        self.n_spurious = 2
        self.core_len = core_len
        self.core_func = core_func
        self.batch_size = batch_size
        self.c = c
        self.sample_num = sample_num
        self.limit_iteration = math.ceil(sample_num/batch_size)
        self.spurious_len = spurious_len
        self.spurious_func = spurious_func
        if c == 1:
            self.n_groups = 2
        else:
            self.n_groups = 2 * self.n_spurious

    def __iter__(self):
        sampled_num = 0
        for i in range(self.limit_iteration):
            if sampled_num + self.limit_iteration > self.sample_num:
                need_batch_size = self.sample_num - sampled_num
            else:
                need_batch_size = self.batch_size
            core_x = generate_random_x(self.core_len, sample_num=need_batch_size, F_2=False).to(self.device)
            core_y = self.core_func(core_x)            
            #same 1, diff 0
            same_or_diff_y = torch.rand(need_batch_size) < self.c
            spurious_y = torch.zeros(need_batch_size).to(self.device)
            same_exist = torch.any(same_or_diff_y)
            diff_exist = torch.any(~same_or_diff_y)
            if same_exist:
                spurious_y[same_or_diff_y==1] = core_y[same_or_diff_y==1]
            if diff_exist:
                spurious_y[same_or_diff_y==0] = -1 * core_y[same_or_diff_y==0]
            spurious_x = torch.zeros((need_batch_size, self.spurious_len)).to(self.device)  
            #print("spurious_y: ", spurious_y.sjape)
            pos_need = (spurious_y==1).sum()
            neg_need = need_batch_size - pos_need
            if self.sampling_method == "on_request":
                spurious_pos_x, spurious_neg_x = get_enough_pos_neg_samples(self.spurious_func, self.spurious_len,
                                                                            pos_need, neg_need, batch_size=2*self.batch_size, device=self.device)
            if pos_need > 0:
                if self.sampling_method == "pure":
                    spurious_x[spurious_y== 1]  = sample(self.spurious_pos_x, pos_need, replacement=True)
                elif self.sampling_method == "on_request":
                    spurious_x[spurious_y== 1] = spurious_pos_x[:pos_need]
            if neg_need > 0:
                if self.sampling_method == "pure":
                    spurious_x[spurious_y==-1]  = sample(self.spurious_neg_x, neg_need, replacement=True)
                elif self.sampling_method == "on_request":
                    spurious_x[spurious_y==-1] = spurious_neg_x[:neg_need]
            spurious_y[spurious_y==-1] = 0
            core_y[core_y==-1] = 0
            core_y = core_y.to(torch.int64)
            if self.c != 1:
                G = core_y * self.n_spurious + spurious_y
            elif self.c == 1:
                G = core_y.clone()            
            x = torch.concat([core_x, spurious_x], dim=1)
            yield x, core_y, G, spurious_y
        
    def __len__(self):
        return self.sample_num
    
def get_enough_pos_neg_samples(boolean_func, x_len, pos_required, neg_required, batch_size, device="cpu"):
    pos_x = []
    neg_x = []
    cur_pos_sample = 0
    cur_neg_sample = 0
    while cur_pos_sample<pos_required or cur_neg_sample<neg_required:
        x = generate_random_x(x_len, sample_num=batch_size, F_2=False).to(device)
        y = boolean_func(x)
        pos_x.append(x[y==1])
        neg_x.append(x[y==-1])
        cur_pos_sample += (y==1).sum()
        cur_neg_sample += (y==-1).sum()
    return torch.concat(pos_x), torch.concat(neg_x)

class SpuriousBooleanFinite(Dataset):
    def __init__(self, core_len, spurious_len, core_func, spurious_func, c, sample_num, 
                 batch_size=64, sampling_method="on_request", bypass_spurious_bias_check=False, device="cpu") -> None:

        self.sampling_distribution = SpuriousBooleanSampling(core_len=core_len, spurious_len=spurious_len, core_func=core_func,
                                                             spurious_func=spurious_func, c=c, sample_num=sample_num,
                                                             batch_size=batch_size, sampling_method=sampling_method, bypass_bias_check=bypass_spurious_bias_check,
                                                             device=device)
        self.device = device
        x_list = []
        core_y_list = []
        G_list = []
        spurious_y_list = []
        self.n_spurious = 2
        for x, core_y, G, spurious_y in self.sampling_distribution:
            x_list.append(x)
            core_y_list.append(core_y)
            G_list.append(G)
            spurious_y_list.append(spurious_y)
        self.X = torch.concat(x_list)
        self.core_y = torch.concat(core_y_list)
        self.G = torch.concat(G_list)
        self.spurious_y = torch.concat(spurious_y_list)
        self.n_classes = 2
        if c == 1:
            self.n_groups = 2
        else:
            self.n_groups = 2 * self.n_spurious

    def __getitem__(self, idx):
        return self.X[idx], self.core_y[idx], self.G[idx], self.spurious_y[idx]

    def __len__(self):
        return self.sampling_distribution.sample_num

    def get_group_counts(self):
        unique_values, counts = torch.unique(self.G, return_counts=True)
        return dict(zip(unique_values.tolist(), counts.tolist()))
    
    
class DominioImage(Dataset):
    #we are going to assume we have two indexable dataset and at each time their ordering is fixed at every initalization
    #core image is always on the top
    def __init__(self, core_dataset, spurious_dataset, confounder_strength, sample_size=None, 
                 unique_spurious=False, concat_dim=2, core_bias_check=True):
        super().__init__()
        self.concat_dim = concat_dim
        max_size = len(core_dataset)
        if not sample_size:
            sample_size = len(core_dataset)
        print(f"sample_size: {sample_size}, dataset_max_size: {max_size}")
        assert sample_size <= max_size
        self.sample_size = sample_size
        self.core_dataset = core_dataset
        self.spurious_dataset = spurious_dataset
        self.concat_dim = concat_dim
        self.confounder_strength = confounder_strength
        assert core_dataset[0][0].shape == spurious_dataset[0][0].shape
        
        core_x_index = torch.arange(len(core_dataset))[:sample_size]
        spurious_x_index = torch.arange(len(spurious_dataset))
        core_y = torch.concat([y for x,y in core_dataset])[:sample_size]
        spurious_y = torch.concat([y for x,y in spurious_dataset])
        if core_bias_check == True:
            ratio_0 = sum(core_y==0)/len(core_y)
            ratio_1 = 1 - ratio_0
            print(f"Bias Check for Core Y: 0: {ratio_0}, 1:{ratio_1}")
            if abs(ratio_0 - ratio_1) >= 0.2:
                print("The bias is too high for core function. You may need check this.")
        core_x_index, core_y, spurious_x_index, spurious_y = mix_spurious_correlated_arrays(core_x_index, core_y,
                                                                                            [spurious_x_index], [spurious_y],
                                                                                            confounder_strength, unique_spurious=unique_spurious)
        self.core_x_index, self.spurious_x_index, self.core_y, self.spurious_y = core_x_index, spurious_x_index, core_y, spurious_y
        self.n_spurious = 2
        self.n_classes = 2
        if confounder_strength == 1:
            self.n_groups = 2
        else:
            self.n_groups = 2 * self.n_spurious
    
    def __getitem__(self, index):
        core_x, core_y = self.core_dataset[self.core_x_index[index]]
        spurious_x, spurious_y = self.spurious_dataset[self.spurious_x_index[index]]
        if self.confounder_strength != 1:
            G = core_y * self.n_spurious + spurious_y
        elif self.confounder_strength == 1:
            G = core_y.clone()
        x = torch.concat([core_x, spurious_x], dim=self.concat_dim)
        return x, core_y, G, spurious_y 
        
    def __len__(self):
        return self.sample_size
    
# class SubsetDataset(Dataset):
#     def __init__(self, full_dataset, subset_indices):
#         self.full_dataset = full_dataset
#         self.subset_indices = subset_indices
#         self.n_groups = full_dataset.n_groups
#     def __len__(self):
#         return len(self.subset_indices)
    
#     def __getitem__(self, idx):
#         return self.full_dataset[self.subset_indices[idx]]

class Spuco2Ours(Dataset):
    def __init__(self, spuco_dataset):
        super().__init__()
        self.dataset = spuco_dataset
        self.n_groups = spuco_dataset.num_spurious * spuco_dataset.num_classes
        self.n_spurious = spuco_dataset.num_spurious
        self.n_labels = spuco_dataset.num_classes
        self.data = spuco_dataset.data.X
        self.labels = torch.tensor(spuco_dataset.data.labels)
        self.groups = torch.tensor(spuco_dataset.data.labels) * torch.tensor(spuco_dataset.num_spurious) \
            + torch.tensor(spuco_dataset.data.spurious)
        self.spurious = torch.tensor(spuco_dataset.data.spurious)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        x, label = self.dataset[index]
        g, p = self.groups[index], self.spurious[index]
        return x,label, g, p

class SpuriousCelebA(Dataset):
    def __init__(self, basedir, split="train", label="Blond_Hair", spurious="Male"):
        super().__init__()
        augmented_transforms = Compose([
            RandomResizedCrop((224, 224), scale=(0.7, 1.0)),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        default_transforms = Compose([
                    Resize((256, 256)),
                    CenterCrop((224, 224)),
                    ToTensor(),
                    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
        if split == "train":
            transform = augmented_transforms
        else:
            transform = default_transforms
        self.transform = transform
        self.dataset = torchvision.datasets.CelebA(root=basedir, download=True, split=split, transform=transform)
        self.n_spurious = 2
        self.n_groups = 4
        self.n_labels = 2
        self.split = split
        
        self.labels = torch.tensor(self.dataset.attr[:,  self.dataset.attr_names.index(label)])
        self.spurious = torch.tensor(self.dataset.attr[:,  self.dataset.attr_names.index(spurious)])
        self.groups = self.labels * 2 + self.spurious
    def __len__(self):
        return len(self.dataset)

    def get_attr_names(self):
        return self.dataset.attr_names

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        y,g,p = self.labels[idx], self.groups[idx], self.spurious[idx]
        return img,y,g,p
    
class WaterBirdsDataset(Dataset):
    # split take values from ["train", "val", "test"]
    # dataset_type takes values from ["combined", "places", "birds"]
    def __init__(self, basedir, confounder_strength, splits=["train","test"], transform=None):
        super().__init__()
        try:
            split_i = [["train", "val", "test"].index(split) for split in splits]
        except ValueError:
            raise (f"Unknown split {splits} or Unknown type of dataset")
        
        dataset_name = f"waterbirds_c{int(confounder_strength*100)}"
        dataset_dir = os.path.join(basedir, dataset_name)
        if not check_dataset_completed(dataset_dir):
            place_dir = os.path.join(basedir, "waterbird_places")
            cub_dir =  os.path.join(basedir, "CUB_200_2011")
            original_construct_waterbirds_dataset(basedir, cub_dir = cub_dir, places_dir=place_dir, confounder_strength=confounder_strength,
                                                  val_frac=0.2, dataset_name=dataset_name)
        metadata_df = pd.read_csv(os.path.join(dataset_dir, "metadata.csv"))
        print("dataset total len: ", len(metadata_df))
        # print(len(metadata_df))
        self.metadata_df = metadata_df[metadata_df["split"].isin(split_i)]
        print("dataset size after split: ", len(self.metadata_df))
        # print(len(self.metadata_df))
        self.basedir = basedir
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.labels = self.metadata_df['y'].values
        self.spurious = self.metadata_df['place'].values
        self.labels = torch.tensor(self.labels)
        self.spurious = torch.tensor(self.spurious)
        self.n_labels = 2
        self.groups = self.labels * 2 + self.spurious
        self.n_spurious = 2
        self.n_groups = self.n_labels * self.n_spurious
        # group_array = y * places + places which classfy the points into different disjoint groups
        # group 0: y = 0 place = 0 -> landbird on land
        # group 1: y = 0 place = 1 -> landbird on water
        # group 2: y = 1 place = 0 -> waterbird on land
        # group 3: y = 1 place = 1 -> waterbird on water
        self.n_groups = 4
        # self.group_counts = (
        #         torch.arange(self.n_groups).unsqueeze(1) == torch.from_numpy(self.group_array)).sum(1).float()
        # self.y_counts = (
        #         torch.arange(self.n_labels).unsqueeze(1) == torch.from_numpy(self.y_array)).sum(1).float()
        # self.p_counts = (
        #         torch.arange(self.n_places).unsqueeze(1) == torch.from_numpy(self.p_array)).sum(1).float()
        self.filename_array = self.metadata_df['img_filename'].values
        self.data = []
        for filename in tqdm(self.filename_array):
            img_path = os.path.join(self.dataset_dir, filename)
            self.data.append(Image.open(img_path).convert('RGB'))

    def __len__(self):
        return int(len(self.metadata_df))

    def __getitem__(self, idx):
        y = self.labels[idx]
        g = self.groups[idx]
        p = self.spurious[idx]
        img = self.data[idx]
        if self.transform:
            img = self.transform(img)
        return img, y, g, p
    