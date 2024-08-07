﻿# Boolean_Spurious
This is the synthetic dataset introduced in https://arxiv.org/abs/2403.03375. It allows fine-grain control on spurious correlation strength, spurious and core feature hardness/complexity. The codebase currently support parity and staircase functions. It is easy for an user to define other boolean function and we provide a demo notebook to show how to use the dataset. 

# Hard Staircase Benchmark
An important agument we made in our paper is that almost all previously proposed debiasing algorithm failed on the "Hard Staircase" setting. Here we specify the parameters for creating "Hard Staircase" dataset.
$\lambda=0.9$, number of samples=60000, $deg(f_s)=10$, $deg(f_c)=14$, $|x_s| = 20$, $|x_c| = 14$. 

# A note on using user-defined biased boolean function

It should be noted that the distribution won't match our formulation in the paper exactly if the user-defined functions are biased (if both functions are unbiased then they are equivalent to the paper formulation). We implement the dataset in this way for efficiency concern. We will update the codebase to match our paper formulation soon. 
