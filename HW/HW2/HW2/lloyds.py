"""
UCSD ECE253: Digital Image Processing
Professor: Mohan Trivedi

Author: You-Yi Jau
Date: 2019.10.14
"""



import numpy as np
from numpy.linalg import norm

def lloyds(training_set, ini_codebook, tol=1e-7, plot_flag=False):
    """
    training_set: np [N, 1]
    ini_codebook: int (number of partitions)
    """
    ## init codebook
    assert len(ini_codebook)==1 # only support that
    if len(ini_codebook) == 1:
        ini_codebook = ini_codebook[0]
        assert ini_codebook >= 1, "invalid ini_codebook"
        min_training = training_set.min()
        max_training = training_set.max()
        ter_condition_2 = np.spacing(1) * max_training
        
        int_training = (max_training - min_training)/ini_codebook
        if int_training <= 0:
            print('The input training set is not valid because it has only one value.')
        codebook = np.linspace(min_training+int_training/2, max_training-int_training/2, ini_codebook);
    # print(f"codebook: {codebook}")

    ## init partition
    partition = (codebook[1 : ini_codebook] + codebook[0 : ini_codebook-1]) / 2
    def quantiz(training_set, partition, codebook):
        # print(f"quan - partition: {partition}, codebook: {codebook}")
        ## compute index
        indx = np.zeros((training_set.shape))
        # print(f"indx: {indx.shape}")
        for i in range(len(partition)):
            indx = indx + (training_set > partition[i])
        ## compute distor
        distor = 0
        for i in range(len(codebook)):
            distor += np.linalg.norm(training_set[indx == i] - codebook[i], 2)
        distor = distor / training_set.shape[0]
        # print(f"quantiz-distor {distor}")
        return indx, distor
    
    def get_rel_distortion(distor, last_distor, ter_condition_2):
        if distor > ter_condition_2:
            rel_distor = abs(distor - last_distor)/distor
        else:
            rel_distor = distor
        return rel_distor
    
    index, distor = quantiz(training_set, partition, codebook)
    last_distor = 0
    # rel_distor = abs(distor - last_distor)/distor
    rel_distor = get_rel_distortion(distor, last_distor, ter_condition_2)
    count = 0
    while (rel_distor > tol) and (rel_distor > ter_condition_2):
        # computer x_hat
        ## handle boundary condition
        partition_aug = np.concatenate((np.array([min_training]), partition, np.array([max_training])))
        # print(f"partition: {partition}")
        # print(f"partition_aug: {partition_aug}")
        for i in range(len(partition_aug)-1):
            part_set = training_set[np.logical_and(training_set>=partition_aug[i], training_set<partition_aug[i+1]) ]
            if len(part_set) > 0:
                codebook[i] = part_set.mean()
            else:
                codebook[i] = (partition_aug[i] + partition_aug[i+1])/2
        
        # update t_hat: codebook
        partition = (codebook[1 : ini_codebook] + codebook[0 : ini_codebook-1]) / 2
        # print(f"count: {count}, partition: {partition}, codebook: {codebook}")
        # quantize again
        last_distor = 0 + distor
        index, distor = quantiz(training_set, partition, codebook)
        
        # get distortion

        # print(f"distor: {distor}, last_distor, {last_distor}, rel_distor: {rel_distor}")
        rel_distor = get_rel_distortion(distor, last_distor, ter_condition_2)
        count += 1
        # print(f"distor: {distor}, rel_distor: {rel_distor}")

    return partition, codebook
