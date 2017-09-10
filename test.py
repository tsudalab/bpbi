# -*- coding: utf-8 -*-

import numpy as np
import sys
import pandas
import bai_algorithms

def open_with_pandas(filename):
    df = pandas.read_csv('data/'+filename)
    data = df.values
    rmsd = data[:,1]
    data = data[:,2:22]

    return rmsd, data


def calc_inlist(v,l):
    if v in l:
        return 1
    else:
        return 0
def calc_hit_ratio(data, true_pose):
    return np.mean(map(lambda x:calc_inlist(x, true_pose), data))

def finish_perm(perms, arm):
    return True if len(perms[arm])==0 else False
    
def check_pose(estimated_pose, rmsd):
    if rmsd[estimated_pose] < 2.0:
        return 'correct pose (< 2.0 Å)'
    else:
        return 'incorrect pose (>= 2.0 Å)'


    



if __name__ == "__main__":
    #Select ligand ID. (2R3J-SCJ, 1OI9-N20, 1KE6-LS2, 3DDQ-RRC, 2WYG-461, 2J94-G15, 2VCI-2GJ, 3VHD-VHE)
    ligand_id = '2J94-G15'
    
    
    #Load RMSD and 20 binding free energeis estimated by MM-PBSA of each binding pose.
    rmsd, data = open_with_pandas(ligand_id + '.csv')
    

    
    
    #----Explore a binding pose using Uniform Sampling
    #Paramter: number of budget
    budget = 40
    best_pose, means, nums = bai_algorithms.uniform_sampling(data, budget = budget)
    print '#### Uniform Sampling ####'
    print 'best pose ID:', best_pose, ', estimated means:',  means, ', pulled arms:', nums    
    #Check the estimated pose is correct (<2.0 Å) or not.
    print check_pose(best_pose, rmsd)
     
    
    
    #----Explore a binding pose using UCB(p) sampling
    #Paramter: number of budget, exploration parameter p
    budget = 30
    p = 8
    best_pose, means, nums = bai_algorithms.UCBp_sampling(data, budget = budget, p = p)
    #Check the result    
    print '#### UCB(p) p =', str(p), ' ####'
    print 'best pose ID:', best_pose, ', estimated means:',  means, ', pulled arms:', nums    
    #Check the estimated pose is correct (<2.0 Å) or not.
    print check_pose(best_pose, rmsd)
    

    
    #----Explore a binding pose using UCB-E sampling
    #Paramter: number of budget, exploration parameter c
    budget = 30
    c = 32
    best_pose, means, nums = bai_algorithms.UCBE_sampling(data, budget = budget, C = c)
    #Check the estimated pose is correct (<2.0 Å) or not.
    print '#### UCB-E c =', str(c), ' ####'
    print 'best pose ID:', best_pose, ', estimated means:',  means, ', pulled arms:', nums    
    #Check the estimated pose is correct (<2.0 Å) or not.
    print check_pose(best_pose, rmsd)
    
    
    #----Explore a binding pose using Successive Rejection
    #Paramter: number of budget
    budget = 30
    best_pose, means, nums = bai_algorithms.SR(data, budget = budget)
    #Check the estimated pose is correct (<2.0 Å) or not
    print '#### SR ####'
    print 'best pose ID:', best_pose, ', estimated means:',  means, ', pulled arms:', nums    
    #Check the estimated pose is correct (<2.0 Å) or not.
    print check_pose(best_pose, rmsd)
    
     
    #----Explore a binding pose using UGapE sampling
    #Paramter: number of budget, exploration parameter a
    budget = 30
    a = 0.25
    best_pose, means, nums = bai_algorithms.UGapE(data, budget = budget, a = a)
    #Check the estimated pose is correct (<2.0 Å) or not.
    print '#### UGapE a =', str(a), ' ####'
    print 'best pose ID:', best_pose, ', estimated means:',  means, ', pulled arms:', nums    
    #Check the estimated pose is correct (<2.0 Å) or not.
    print check_pose(best_pose, rmsd)


    #----Explore a binding pose using UCB-E auto                                                                                                               
    #Paramter: number of budget                                                                                                          
    budget = 40
    best_pose, means, nums = bai_algorithms.UCBE_adaptive(data, budget = budget)
    #Check the estimated pose is correct (<2.0 Å) or not.                                                                                                         
    print '#### UCB-E adaptive ####'
    print 'best pose ID:', best_pose, ', estimated means:',  means, ', pulled arms:', nums
    #Check the estimated pose is correct (<2.0 Å) or not.                                                                                                         
    print check_pose(best_pose, rmsd)
    
    #----Explore a binding pose using UGapE auto                                                                                                             
    #Paramter: number of budget                                                                                                   
    budget = 40
    best_pose, means, nums = bai_algorithms.UGapE_adaptive(data, budget = budget)
    #Check the estimated pose is correct (<2.0 Å) or not.                                                                                                 
    print '#### UGapE a =', str(a), ' ####'
    print 'best pose ID:', best_pose, ', estimated means:',  means, ', pulled arms:', nums
    #Check the estimated pose is correct (<2.0 Å) or not.                                                                                             
    print check_pose(best_pose, rmsd)    
    
    
    #----Calculation example of prob. of correct pose prediction using uniform sampling 
    result = [bai_algorithms.uniform_sampling(data, 40)[0] for i in range(30000)]    
    true_pose_index = filter(lambda x:rmsd[x]<=2.0 , range(20))
    print 'hit_ratio', calc_hit_ratio(result, true_pose_index)   
    

    
    
    
    
    
    
