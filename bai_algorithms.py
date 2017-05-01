# -*- coding: utf-8 -*-
import numpy as np
import copy as cp
import sys


def finish_perm(perms, arm):
    return True if len(perms[arm])==0 else False


def uniform_sampling(data, budget):
    
    if budget % 20 != 0 or budget < 0:
        sys.stderr.write('Error (budget should be multiples of arms) !')
        
    perms = [list(np.random.permutation(20)[0:budget]) for i in range(20)] 
    
    
    #Initialization
    means = []
    nums = []
    for i in range(20):
        run = perms[i].pop(0)
        means.append(-data[i][run])
        nums.append(1)    
    
    #Exploration loop
    for round in range(20, budget):
        #Select an arm               
        arm = round % 20
        
        #Check the rest data of the arm.
        if finish_perm(perms, arm):
            break

        #Get a reward from the arm
        nums[arm] += 1
        run = perms[arm].pop(0)
        means[arm] = (means[arm]*(nums[arm]-1) - data[arm][run])/nums[arm]
            
    return np.argmax(means), means, nums
    
    
   

def calc_UCBp(t, mean, num, p):
    #print len(l)
    if num>0:
        return mean + np.sqrt(p*np.log(t)/num)
    else:
        return 10**10

    
def UCBp_sampling(data, budget, p):
    
    perms = [list(np.random.permutation(20)) for i in range(20)]            
      
    #Initialization
    means = []
    nums = []
    for i in range(20):
        run = perms[i].pop(0)
        means.append(-data[i][run])
        nums.append(1)        
    #Exploration loop
    for round in range(20, budget):
        #Select an arm        
        ucbs = [calc_UCBp(round, means[i], nums[i], p) for i in range(20)]
        maxpos = np.argmax(ucbs)
        
        ##Check the rest data of the arm.
        if finish_perm(perms, maxpos):
            break

        #Get a reward from the arm
        run = perms[maxpos].pop(0)
        nums[maxpos] += 1
        means[maxpos] = (means[maxpos]*(nums[maxpos]-1) - data[maxpos][run])/nums[maxpos]
        
        
    return np.argmax(means), means, nums
    
 
    
def calc_UCBE(mean, num, C, H1, budget):
    #print len(l)
    if num>0:
        #a = C*budget/H1
        return mean + np.sqrt(C/num)
    else:
        return 10**10

def UCBE_sampling(data, budget, C):
    
    H1 = 1 
    
    perms = [list(np.random.permutation(20)) for i in range(20)]            
      
    #Initialization
    means = []
    nums = []
    for i in range(20):
        run = perms[i].pop(0)
        means.append(-data[i][run])
        nums.append(1)        
    #Exploration loop
    for round in range(20, budget):
        #Select an arm         
        ucbs = [calc_UCBE(means[i], nums[i], C, H1, budget) for i in range(20)]
        maxpos = np.argmax(ucbs)
        
        #Check the rest data of the arm.
        if finish_perm(perms, maxpos):
            break
        
        #Get a reward from the arm
        run = perms[maxpos].pop(0)
        nums[maxpos] += 1
        means[maxpos] = (means[maxpos]*(nums[maxpos]-1) - data[maxpos][run])/nums[maxpos]
       
            
    return np.argmax(means), means, nums
    
    

def blog(k):
    return 0.5 + sum([(1./i) for i in range(2, k+1)])

def nk(n, K, k, blogK): #K: number of arms
    if k == 0:
        return 0
    else:
        return np.ceil((n - k)/(blogK*(K+1-k)))


def SR(data, budget):

    blogK = blog(20)
    
    means = [[] for i in range(20)]
    nums = [0]*20
    Arms = range(20) 
    perms = [list(np.random.permutation(20)) for i in range(20)] 
        
    for k in range(1, 20): #budget_unit回一つづつ抜いていく
        for i in Arms:
            for x in range(int(nk(budget, 20, k,  blogK) - nk(budget, 20, k-1, blogK))):
                #Check the rest data of the arm.
                if finish_perm(perms, i):
                    break               
                
                run = perms[i].pop(0)
                means[i].append(-data[i][run])
                nums[i] += 1
        minarm_index = np.argmin([np.mean(means[i]) for i in Arms])
        Arms.pop(minarm_index)
    
    return Arms[0], map(np.mean, means), nums    
        

            

def beta(s, budget, K, a):
    return np.sqrt(a*(budget - K) / s) 

def U(i, t, budget, pulls, a):
    return np.mean(pulls[i]) + beta(len(pulls[i]), budget, 20, a)

def L(i, t, budget, pulls, a):
    return np.mean(pulls[i]) - beta(len(pulls[i]), budget, 20, a)

def B(i, t, budget, pulls, a, K):
    list_woi = range(20)
    list_woi.pop(i)
    return np.max([U(j, t, budget, pulls, a) - L(i, t, budget, pulls, a)  for j in list_woi])

def calc_B(i, U_l, L_l):
    list_woi = range(20)
    list_woi.pop(i)
    return np.max([U_l[j] - L_l[i] for j in list_woi])


def UGapE(data, budget, a):
    perms = [list(np.random.permutation(20)) for i in range(20)]            
      
    #Initialization
    pulls = [[] for i in range(20)]
    nums = []
    for i in range(20):
        run = perms[i].pop(0)
        pulls[i].append(-data[i][run])
        nums.append(1)       
        
    #Exploration loop
    for round in range(20, budget):
        #Select an arm  
        mean_list = [np.mean(pulls[i]) for i in range(20)]
        beta_list = [beta(len(pulls[i]), round, 20, a) for i in range(20)]
        U_list = [mean_list[i] + beta_list[i] for i in range(20)]
        L_list = [mean_list[i] - beta_list[i] for i in range(20)]
        B_list = [calc_B(i, U_list, L_list) for i in range(20)]
                       
        
        
        J_t = np.argmin(B_list)
        list_woJt = range(20)
        list_woJt.pop(J_t)
        u_t = np.argmax([U_list[i] for i in list_woJt])
        
                       
        if u_t>=J_t:
            u_t += 1
        
        l_t = J_t 
        I_t = [l_t, u_t][np.argmax([beta_list[i] for i in [l_t, u_t]])]                

        #Check the rest data of the arm.
        if finish_perm(perms, I_t):
            break
        run = perms[I_t].pop(0)
        nums[I_t] += 1
        pulls[I_t].append(-data[I_t][run])
        

    
    #Select final result based on Beta
    mean_list = [np.mean(pulls[i]) for i in range(20)]
    beta_list = [beta(len(pulls[i]), budget, 20, a) for i in range(20)]
    U_list = [mean_list[i] + beta_list[i] for i in range(20)]
    L_list = [mean_list[i] - beta_list[i] for i in range(20)]
    B_list = [calc_B(i, U_list, L_list) for i in range(20)]
    
    return np.argmin(B_list), mean_list, nums    
    
    
    #Select final result based on mean values
    #return np.argmax(mean_list), mean_list, nums
    
