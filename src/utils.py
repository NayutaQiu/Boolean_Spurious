import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import itertools
from functools import partial
import random
import pandas as pd
from torch import nn
import os
import warnings
from sklearn.exceptions import ConvergenceWarning
from collections import OrderedDict, defaultdict

def sample_index(x, sample_size, replacement=False):
    if not replacement:
        assert sample_size <= len(x)
        return torch.randperm(len(x))[:sample_size]
    else:
        return torch.randint(0, len(x), size=(sample_size,))


def sample(x, sample_size, replacement=False):
    if replacement == "auto":
        replacement = sample_size > len(x)
    if replacement == True:
        return x[torch.randint(0, len(x), size=(sample_size,))]
    else:
        assert sample_size <= len(x)
        return x[torch.randperm(len(x))][:sample_size]


def generate_random_func(n, k, F2=True):
    mapping = generate_random_x(1,sample_num= 2**k, F_2=F2)
    subset = tuple(random.sample(range(n), k=k))
    return RandomFunc(mapping, subset, F2=F2), subset

def generate_random_func_with_subset(subset, F2=True):
    k = len(subset)
    mapping = generate_random_x(1,sample_num= 2**k, F_2=F2)
    return RandomFunc(mapping, subset, F2=F2)

class BooleanFunc(object):
    pass

class RandomFunc(BooleanFunc):
    def __init__(self, mapping, subset, F2=True):
        self.F2 = F2
        self.mapping = mapping.cuda()
        self.subset = subset
    def __call__(self, x_s):
        #x_s = x_s.cpu().type(torch.long)
        if not self.F2:
            x_s = torch.where(x_s == -1, 0, x_s)
        indices = torch.zeros(x_s.shape[0], dtype=torch.long).cuda()
        for c, i in enumerate(self.subset):
            indices += (x_s[:,i] * 2**c).type(torch.long)
        return self.mapping[indices].squeeze().cuda()

def generate_bit_vectors(n):
    # Generate a tensor with all possible bit values
    bit_values = torch.tensor([0, 1])

    # Compute the Cartesian product of bit_values to generate all bit vectors
    bit_vectors = torch.cartesian_prod(*[bit_values] * n)

    return bit_vectors
    
def generate_random_x(length, sample_num=None, F_2=True, unique=False):
    if sample_num:
        x = torch.randint(low=0, high=2, size=(sample_num, length), dtype=torch.float32)
        if unique:
            x = torch.unique(x,dim=0)
    else:
        values = list(itertools.product([0, 1], repeat=length))
        x = torch.tensor(values, dtype=torch.float32)
    if not F_2:
        x[x==0] = -1
    return x

def majority(x_s, subset=[]):
    assert len(subset) % 2 == 1
    return torch.sign(torch.sum(x_s[:, subset], dim=1))

def generate_majority_func(subset):
    return partial(majority, subset=subset)

def generate_random_majority(feature_len, subset_size):
    assert 0<subset_size<=feature_len
    subset = tuple(random.sample(range(feature_len), k=subset_size))
    return generate_majority_func(subset), [subset]

def generate_parity_func(subset):
    return partial(parity, subset=subset)

def generate_fixed_parity_func(func_degree):
    subset=tuple(range(func_degree))
    return partial(parity, subset=subset), [subset]

def generate_random_parity_func(feature_len, subset_size):
    assert 0<subset_size<=feature_len
    subset = tuple(random.sample(range(feature_len), k=subset_size))
    return generate_parity_func(subset), [subset]

def get_sample_space(model, feature_len, sample_num=None, batch_size=64, F_2=True,
                     unique=False):
    feature_space = generate_random_x(feature_len, sample_num = sample_num, F_2=F_2, unique=unique) 
    y_s = batch_forward(model, feature_space, batch_size=batch_size, F_2=F_2)
    return feature_space, y_s

def batch_forward(model, x, batch_size=64, F_2=True, R=False, device="cpu"):
    if isinstance(model, torch.nn.Module):
        return model_batch_forward(model, x, batch_size=batch_size, F_2=F_2, R=R, device=device)
    else:
        return func_batch_forward(model, x, batch_size=batch_size, device=device)


def model_batch_forward(model, x, batch_size=64, F_2=True, R=False, device="cpu"):
    model = model.to(device)
    feature_space_dataset = TensorDataset(x)
    feature_space_loader = DataLoader(feature_space_dataset, batch_size=batch_size, shuffle=False)
    y_s = []
    model.eval()
    with torch.no_grad():
        for x in feature_space_loader:
            x = x[0].to(device)
            if R:
                y = model(x)
            elif not F_2:
                y = model(x).argmax(dim=1)
                y[y==0] = -1
            else:
                y = model(x)
            y_s.append(y)
    y_s = torch.concat(y_s)
    return y_s


def func_batch_forward(func, x, batch_size=64, device="cpu"):
    feature_space_dataset = TensorDataset(x)
    feature_space_loader = DataLoader(feature_space_dataset, batch_size=batch_size, shuffle=False)
    y_s = []
    with torch.no_grad():
        for x in feature_space_loader:
            x = x[0].to(device)
            y = func(x)
            y_s.append(y)
    y_s = torch.concat(y_s)
    return y_s

def sample_bit_vectors(n, k):
    """
    Sample n bit vectors of length k with an equal distribution of 0s and 1s.

    Arguments:
    - n: The number of bit vectors to sample.
    - k: The length of each bit vector.

    Returns:
    - A tensor of shape (n, k) containing the sampled bit vectors.
    """

    # Sample the bit vectors
    bit_vectors = torch.randint(low=0, high=2, size=(n, k))

    return bit_vectors

def fourier_weight_estimate(model, subset, n, est_num = 100, F_2=True, R=False, device="cpu"):
    """
    model: any model that support model(tensor) = {-1,1} 
    subset: contains the index of variables for parity taking values from [0, n-1]. None represent fourier weight for the empty set.
    """
    #draw sample x
    x_s = generate_random_x(n, est_num, F_2=F_2)
    if not subset:
        parity_S = torch.ones(est_num)
    else:
        assert all([0 <= i and i < n for i in subset])
        parity = generate_parity_func(subset)
        parity_S = func_batch_forward(parity, x_s)
    parity_S = parity_S.type(torch.float).to(device)
    pred_y = batch_forward(model, x_s, F_2=F_2, R=R).type(torch.float).squeeze()
    #print(pred_y, parity_S)
    return (pred_y @ parity_S) / est_num


def fourier_weight(model, subset, n, F_2=True, R=False, device="cpu"):
    """
    model: any model that support model(tensor) = {-1,1} 
    subset: contains the index of variables for parity taking values from [0, n-1]. None represent fourier weight for the empty set.
    """
    #draw sample x
    x_s = generate_random_x(n, None, F_2=F_2)
    if not subset:
        parity_S = torch.ones(2**n)
    else:
        assert all([0 <= i and i < n for i in subset])
        parity = generate_parity_func(subset)
        parity_S = func_batch_forward(parity, x_s)
    parity_S = parity_S.type(torch.float).to(device)
    pred_y = batch_forward(model, x_s, F_2=F_2, R=R).type(torch.float).squeeze()
    #print(pred_y, parity_S)
    return (pred_y @ parity_S) / 2**n

def correlation_estimate(model, func, n, est_num = 100, F_2=True, starting_index=0, ending_index=None, device="cpu"):
    """
    model: any model that support model(tensor) = {-1,1} 
    subset: contains the index of variables for parity taking values from [0, n-1]. None represent fourier weight for the empty set.
    """
    #draw sample x
    if ending_index == None:
        ending_index = n
    x_s = generate_random_x(n, est_num, F_2=F_2)
    func_y= func(x_s[:, starting_index:ending_index]).to(device)
    func_y = func_y.type(torch.float).to(device)
    pred_y = batch_forward(model, x_s, F_2=F_2, device=device).type(torch.float).squeeze()
    #print(func_y, pred_y)
    return ((pred_y @ func_y) / est_num).cpu().item()

def fourier_dict_records_to_df(fourier_dicts):
    res_df =  pd.DataFrame(fourier_dicts)
    return res_df

    
from itertools import chain, combinations

def powerset(iterable, without_emptyset = False):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    result =  chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    if without_emptyset:
        result.__next__()
    return result

def estimate_fourier_expansion(model, n, sample_num = 100, F_2=True, targets=[], R=False, device="cpu"):
    fourier_dict = dict()
    if not targets:
        targets = powerset(range(n))
    for subset in targets:
        fourier_weight_est = fourier_weight_estimate(model, subset, n, sample_num, F_2=F_2, R=R, device=device)
        fourier_dict[tuple(subset)] = fourier_weight_est.cpu().item()
    return fourier_dict




def parity(x_s, subset=[]):
    if not subset:
        return torch.ones(len(x_s))
    else:
        return torch.prod(x_s[:,subset], dim=1)
    


from sklearn.linear_model import LogisticRegression
def estimate_decoded_fourier_expansion(model, n, sample_num, targets, F_2=True):
    fourier_dict = dict()
    if not targets:
        targets = powerset(range(n))
    for subset in targets:
        fourier_weight_est = fourier_decoded_weight_estimate(model, subset, n, sample_num, F_2=F_2)
        fourier_dict[tuple(subset)] = fourier_weight_est.item()
    return fourier_dict

def fourier_decoded_weight_estimate(model, subset, n, est_num = 1000, F_2=True, unique=False):
    train_propotion = 0.8
    break_index = round(est_num*train_propotion)
    x_s = generate_random_x(n, est_num, unique=unique, F_2=F_2)
    if not subset:
        parity_S = torch.ones(est_num)
    else:
        assert all([0 <= i and i < n for i in subset])
        parity = generate_parity_func(subset)
        parity_S = func_batch_forward(parity, x_s)
    parity_S = parity_S.type(torch.float).cpu()
    embeddings = batch_forward_embedding(model, x_s).cpu()
    training_x, training_y = embeddings[:break_index], parity_S[:break_index]
    testing_x, testing_y = embeddings[break_index:], parity_S[break_index:]
    #print(training_x)
    lr = LogisticRegression()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        lr.fit(training_x, training_y)
    return lr.score(testing_x, testing_y)

def generate_staircase_func(degree, device="cpu"):
    parity_subsets = [tuple(range(i)) for i in range(1, degree+1)]
    func = PolynomialThreshold(parity_subsets, [1 for _ in range(len(parity_subsets))], device=device)
    return func, parity_subsets

def batch_forward_embedding(model, x, batch_size=64, device="cpu"):
    model = model.to(device)
    feature_space_dataset = TensorDataset(x)
    feature_space_loader = DataLoader(feature_space_dataset, batch_size=batch_size, shuffle=False)
    embedding_s = []
    model.eval()
    with torch.no_grad():
        for x in feature_space_loader:
            x = x[0].to(device)
            embedding = model.embedding(x)
            embedding_s.append(embedding)
    emb = torch.concat(embedding_s)
    return emb

def decoded_accuracy_on_func(model, func, n, starting_index=0, ending_index=None, est_num = 1000, F_2=True, unique=False, device="cpu"):
    #Total accuracy
    if ending_index == None:
        ending_index = n
    train_propotion = 0.8
    break_index = round(est_num*train_propotion)
    x_s = generate_random_x(n, est_num, unique=unique, F_2=F_2).to(device)
    y_s = func(x_s[:, starting_index:ending_index])
    y_s = y_s.type(torch.float).cpu()
    embeddings = batch_forward_embedding(model, x_s, device=device).cpu()
    training_x, training_y = embeddings[:break_index], y_s[:break_index]
    testing_x, testing_y = embeddings[break_index:], y_s[break_index:]
    lr = LogisticRegression()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        lr.fit(training_x, training_y)
    return lr.score(testing_x, testing_y)


def decoded_accuracy_on_spurious_dataset(model, dataloader, est_num=None, train_propotion=0.8):
    max_size = len(dataloader.dataset)
    if est_num == None:
        est_num = max_size
    else:
        if est_num > max_size:
            print(f"Warning: required est_num {est_num} > dataset size {max_size}. And it has been set to the dataset size")
            est_num = max_size
            
    embedding_s, core_y_s, group_y_s, spurious_y_s = batch_forward_embedding_spurious_dataset(model, dataloader, sample_size=est_num)
    break_index = round(est_num*train_propotion)
    embedding_s, core_y_s, spurious_y_s = embedding_s.cpu(), core_y_s.cpu().squeeze(), spurious_y_s.cpu().squeeze()
    training_embedding, training_core_y, training_spurious_y = embedding_s[:break_index], core_y_s[:break_index], spurious_y_s[:break_index]
    testing_embedding, testing_core_y, testing_spurious_y = embedding_s[break_index:est_num], core_y_s[break_index:est_num], spurious_y_s[break_index:est_num]
    res_dict = dict()
    lr = LogisticRegression()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        lr.fit(training_embedding, training_core_y)
    res_dict["core"] = lr.score(testing_embedding, testing_core_y)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        lr.fit(training_embedding, training_spurious_y)
    res_dict["spurious"] = lr.score(testing_embedding, testing_spurious_y)
    return res_dict

def accuracy_to_correlation(accuracy):
    accuracy = torch.tensor(accuracy)
    corr = accuracy - (1-accuracy)
    return corr

def batch_forward_prediction_spurious_dataset(model, dataloader, sample_size=None, required_grad=False):
    model = model.cuda()
    pred_s = []
    core_y_s = []
    group_y_s = []
    spurious_y_s = []
    read_size = 0
    model.eval()
    max_size = len(dataloader.dataset)
    if not sample_size:
        sample_size = max_size
    if max_size < sample_size:
        print(f"Warning: sample_size: {sample_size} greater than total_size: {read_size}")
    with torch.no_grad():
        for x, core_y, G, spurious_y in dataloader:
            if read_size >= sample_size:
                break
            x = x.cuda()
            pred = torch.argmax(model(x), dim=1)
            pred_s.append(pred)
            core_y_s.append(core_y)
            group_y_s.append(G)
            spurious_y_s.append(spurious_y)
            read_size += len(core_y)
    pred_s, core_y_s, group_y_s, spurious_y_s = torch.concat(pred_s), torch.concat(core_y_s),torch.concat(group_y_s), torch.concat(spurious_y_s)
    return pred_s, core_y_s, group_y_s, spurious_y_s

def batch_forward_logits_spurious_dataset(model, dataloader, sample_size=None, required_grad=False):
    model = model.cuda()
    pred_s = []
    core_y_s = []
    group_y_s = []
    spurious_y_s = []
    read_size = 0
    model.eval()
    max_size = len(dataloader.dataset)
    if not sample_size:
        sample_size = max_size
    if max_size < sample_size:
        print(f"Warning: sample_size: {sample_size} greater than total_size: {read_size}")
    with torch.no_grad():
        for x, core_y, G, spurious_y in dataloader:
            if read_size >= sample_size:
                break
            x = x.cuda()
            pred = model(x)
            pred_s.append(pred)
            core_y_s.append(core_y)
            group_y_s.append(G)
            spurious_y_s.append(spurious_y)
            read_size += len(core_y)
    pred_s, core_y_s, group_y_s, spurious_y_s = torch.concat(pred_s), torch.concat(core_y_s),torch.concat(group_y_s), torch.concat(spurious_y_s)
    return pred_s, core_y_s, group_y_s, spurious_y_s


def model_correlation_on_spurious_dataset(model, dataloader, est_num=None, train_propotion=0.8):
    max_size = len(dataloader.dataset)
    if est_num == None:
        est_num = max_size
    else:
        if est_num > max_size:
            print(f"Warning: required est_num {est_num} > dataset size {max_size}. And it has been set to the dataset size")
            est_num = max_size
            
    pred_s, core_y_s, group_y_s, spurious_y_s = batch_forward_prediction_spurious_dataset(model, dataloader, sample_size=est_num)
    pred_s, core_y_s, spurious_y_s = pred_s.cpu(), core_y_s.cpu().squeeze(), spurious_y_s.cpu().squeeze()
    pred_s, core_y_s, spurious_y_s = pred_s[:est_num], core_y_s[:est_num], spurious_y_s[:est_num]
    res_dict = dict()
    res_dict["core_accuracy"] = torch.eq(pred_s, core_y_s).float().mean().item()
    res_dict["spurious_accuracy"] = torch.eq(pred_s, spurious_y_s).float().mean().item()
    res_dict["core_correlation"] = 2*res_dict["core_accuracy"] - 1
    res_dict["spurious_correlation"] = 2*res_dict["spurious_accuracy"] - 1
    return res_dict
    
def batch_forward_embedding_spurious_dataset(model, dataloader, sample_size=None):
    model = model.cuda()
    embedding_s = []
    core_y_s = []
    group_y_s = []
    spurious_y_s = []
    read_size = 0
    model.eval()
    max_size = len(dataloader.dataset)
    if not sample_size:
        sample_size = max_size
    if max_size < sample_size:
        print(f"Warning: sample_size: {sample_size} greater than total_size: {read_size}")
    with torch.no_grad():
        for x, core_y, G, spurious_y in dataloader:
            if read_size >= sample_size:
                break
            x = x.cuda()
            embedding = model.embedding(x).cpu()
            embedding_s.append(embedding)
            core_y_s.append(core_y)
            group_y_s.append(G)
            spurious_y_s.append(spurious_y)
            read_size += len(core_y)
    embedding_s, core_y_s, group_y_s, spurious_y_s = torch.concat(embedding_s), torch.concat(core_y_s),torch.concat(group_y_s), torch.concat(spurious_y_s)
    return embedding_s, core_y_s, group_y_s, spurious_y_s

def generate_random_polynomial(n, k_s):
    assert 0 < len(k_s) and len(k_s) <= n
    subsets_all = []
    coefs_all = []
    for i in range(len(k_s)):
        subsets = random.sample(list(itertools.combinations(range(n), i+1)), k=k_s[i])
        coefs = (2 * torch.rand(k_s[i]) - 1).tolist()
        subsets_all.extend(subsets)
        coefs_all.extend(coefs)
        
    return PolynomialThreshold(subsets_all, coefs_all)
def custom_sign(x):
    signed_x = torch.sign(x)
    signed_x[signed_x==0] = 1
    return signed_x
class PolynomialThreshold(BooleanFunc):
    def __init__(self, subsets_all, coefs_all, device="cpu"):
        self.subsets_all = subsets_all
        self.coefs_all = coefs_all
        self.device = device
        self.funcs = []
        for subset in subsets_all:
            func = generate_parity_func(subset)
            self.funcs.append(func)
        self.fourier_dict = dict(zip(subsets_all, coefs_all))
    
    def __call__(self, x):
        res = torch.zeros(len(x)).to(self.device)
        for func, coef in zip(self.funcs, self.coefs_all):
            res += coef * func(x).to(self.device)
        return custom_sign(res)
    
class Majority(BooleanFunc):
    def __init__(self):
        pass
    
    def __call__(self, x):
        return torch.sum(x, dim=1)
    
class Polynomial(BooleanFunc):
    def __init__(self, subsets_all, coefs_all):
        self.subsets_all = subsets_all
        self.coefs_all = coefs_all
        self.funcs = []
        for subset in subsets_all:
            func = generate_parity_func(subset)
            self.funcs.append(func)
        self.fourier_dict = dict(zip(subsets_all, coefs_all))
    
    def __call__(self, x):
        res = torch.zeros(len(x)).cuda()
        for func, coef in zip(self.funcs, self.coefs_all):
            res += coef * func(x).cuda()
        return res
