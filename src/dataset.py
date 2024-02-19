
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.utils.data.sampler import WeightedRandomSampler
from transformers import BertTokenizer
from utils import *
from dataset_utils import *
import math

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
