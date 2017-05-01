# -*- coding: utf-8 -*-
import numpy as np
import sys
import pandas
import bai_algorithms

def open_with_pandas(filename):
    df = pandas.read_csv(filename)
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
    #Select ligand ID. (2R3J-SCJ, 1OI9-N20, 1KE6-LS2, 3DDQ-RRC)
    ligand_id = '2R3J-SCJ'
    
    
    #Load RMSD and 20 binding free energeis estimated by MM-PBSA of each binding pose.
    rmsd, data = open_with_pandas(ligand_id + '.csv')
    

    
    
    #----Explore a binding pose using Uniform Sampling
    #Paramter: number of budget
    best_pose, means, nums = bai_algorithms.uniform_sampling(data, budget = 40)
    print 'best pose ID:', best_pose, ', estimated means:',  means, ', pulled arms:', nums    
    #Check the estimated pose is correct (<2.0 Å) or not.
    print check_pose(best_pose, rmsd)
     
    
    
    #----Explore a binding pose using UCB(p) sampling
    #Paramter: number of budget
    best_pose, means, nums = bai_algorithms.UCBp_sampling(data, budget = 30, p = 8)
    #Check the result    
    print 'best pose ID:', best_pose, ', estimated means:',  means, ', pulled arms:', nums    
    #Check the estimated pose is correct (<2.0 Å) or not.
    print check_pose(best_pose, rmsd)
    

    
    #----Explore a binding pose using UCB-E sampling
    #Paramter: number of budget
    best_pose, means, nums = bai_algorithms.UCBE_sampling(data, budget = 30, C = 32)
    #Check the estimated pose is correct (<2.0 Å) or not.
    print 'best pose ID:', best_pose, ', estimated means:',  means, ', pulled arms:', nums    
    #Check the estimated pose is correct (<2.0 Å) or not.
    print check_pose(best_pose, rmsd)
    
    
    #----Explore a binding pose using Successive Rejection
    #Paramter: number of budget
    best_pose, means, nums = bai_algorithms.SR(data, budget = 30)
    #Check the estimated pose is correct (<2.0 Å) or not.
    print 'best pose ID:', best_pose, ', estimated means:',  means, ', pulled arms:', nums    
    #Check the estimated pose is correct (<2.0 Å) or not.
    print check_pose(best_pose, rmsd)
    
     
    #----Explore a binding pose using UGapE sampling
    #Paramter: number of budget
    best_pose, means, nums = bai_algorithms.UGapE(data, budget = 30, a = 0.25)
    #Check the estimated pose is correct (<2.0 Å) or not.
    print 'best pose ID:', best_pose, ', estimated means:',  means, ', pulled arms:', nums    
    #Check the estimated pose is correct (<2.0 Å) or not.
    print check_pose(best_pose, rmsd)
    
    
    
    
    #----Calculation example of prob. of correct pose prediction using uniform sampling 
    result = [bai_algorithms.uniform_sampling(data, 40)[0] for i in range(30000)]    
    true_pose_index = filter(lambda x:rmsd[x]<=2.0 , range(20))
    print 'hit_ratio', calc_hit_ratio(result, true_pose_index)   
    
    
    
    
    
    
    
    
