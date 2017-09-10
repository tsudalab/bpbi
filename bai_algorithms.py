# -*- coding: utf-8 -*-
import numpy as np
import copy as cp
import sys


def finish_perm(perms, arm):
    return True if len(perms[arm])==0 else False


def uniform_sampling(data, budget):
    K = len(data) #num of arms
    
    if budget % K != 0 or budget < 0:
        sys.stderr.write('Error (budget should be multiples of arms) !')
        
    perms = [list(np.random.permutation(len(data[0]))[0:budget]) for i in range(K)] 
    
    
    #Initialization
    means = []
    nums = []
    for i in range(K):
        run = perms[i].pop(0)
        means.append(-data[i][run])
        nums.append(1)    
    
    #Exploration loop
    for round in range(K, budget):
        #Select an arm               
        arm = round % K
        
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
    if num == 20:
        return - 10**10
    elif 1 <= num:
        return mean + np.sqrt(p*np.log(t)/num)
    else:
        return 10**10

def UCBp_sampling(data, budget, p):
    K = len(data)
    
    perms = [list(np.random.permutation(len(data[0]))) for i in range(K)]            
      
    #Initialization
    means = []
    nums = []
    for i in range(K):
        run = perms[i].pop(0)
        means.append(-data[i][run])
        nums.append(1)        

    #Exploration loop
    for round in range(K, budget):

        #Select an arm        
        ucbs = [calc_UCBp(round, means[i], nums[i], p) for i in range(K)]
        maxpos = np.argmax(ucbs)
        
        
        #Get a reward from the arm
        run = perms[maxpos].pop(0)
        nums[maxpos] += 1
        means[maxpos] = (means[maxpos]*(nums[maxpos]-1) - data[maxpos][run])/nums[maxpos]
        
        
    return np.argmax(means), means, nums
 

    
 
    
def calc_UCBE(mean, num, C, budget):
    
    if num == 20:
        return - 10**10
    elif num>0:
        return mean + np.sqrt(C/num)
    else:
        return 10**10



def UCBE_sampling(data, budget, C = 1):
    K = len(data)
    H1 = 1 
    
    perms = [list(np.random.permutation(len(data[0]))) for i in range(K)]            
      
    #Initialization
    means = []
    nums = []
    for i in range(K):
        run = perms[i].pop(0)
        means.append(-data[i][run])
        nums.append(1)        
    #Exploration loop
    for round in range(K, budget):
        #Select an arm         
        ucbs = [calc_UCBE(means[i], nums[i], C, budget) for i in range(K)]
            
        maxpos = np.argmax(ucbs)
        
        
        #Get a reward from the arm
        run = perms[maxpos].pop(0)
        nums[maxpos] += 1
        means[maxpos] = (means[maxpos]*(nums[maxpos]-1) - data[maxpos][run])/nums[maxpos]
       
            
    return np.argmax(means), means, nums
    

def UCBE_adaptive(data, budget, auto_para = 1):
    K = len(data)
    perms = [list(np.random.permutation(len(data[0]))) for i in range(K)]
    blogK = blog(K)

    #Initialization                                                                                                                                                
    rewards = [[] for i in range(K)]
    nums = [0 for i in range(K)]


    for k in range(0, K):

        if k == 0:
            H_k = K
        else:
            mean_list = [np.mean(rewards[i]) for i in range(K)]
            max_mean = np.max(mean_list)
            delta_list = [max_mean - mean_list[i] for i in range(K)]
            
            delta_list_ordered = sorted(delta_list)

            H_k = np.max([i / (delta_list_ordered[i-1]**2)  for i in range(K - k + 1, K+1)])


        for t in range(tk(budget, K, k, blogK)+1, tk(budget, K, k+1, blogK)+1):
            if t <= 20:
                maxpos = t-1
                run = perms[maxpos].pop(0)
                rewards[maxpos].append(-data[maxpos][run])
                nums[maxpos] += 1
            else:                
                mean_list = [np.mean(rewards[i]) for i in range(K)]
                ucbs = [calc_UCBE(mean_list[i], nums[i], float(budget)/H_k, budget) for i in range(K)]

                maxpos = np.argmax(ucbs)
                run = perms[maxpos].pop(0)
                nums[maxpos] += 1
                rewards[maxpos].append(- data[maxpos][run])

    print 'total pulls', np.sum(nums)
    mean_list = [np.mean(rewards[i]) for i in range(K)]
    return np.argmax(mean_list), mean_list, nums


def blog(k):
    return 0.5 + sum([(1./i) for i in range(2, k+1)])

def nk(n, K, k, blogK): #K: number of arms
    if k == 0:
        return 0
    else:
        return np.ceil((n - k)/(blogK*(K+1-k)))

def tk(n, K, k, blogK):
    if k == 0:
        tk = 0
    elif k == 1:
        tk = K*nk(n, K, k, blogK)
    elif k == K:
        tk = n
    else:
        tk = np.sum([nk(n, K, i, blogK) for i in range(1, k)]) + (K - k + 1)*nk(n, K, i, blogK)
    return int(tk)

def SR(data, budget):
    K = len(data)

    blogK = blog(K)
    
    means = [[] for i in range(K)]
    nums = [0]*K
    Arms = range(K) 
    perms = [list(np.random.permutation(len(data[0]))) for i in range(K)] 
        
    for k in range(1, K): 
        for i in Arms:
            for x in range(int(nk(budget, K, k,  blogK) - nk(budget, K, k-1, blogK))):
                #Check the rest data of the arm.
                if finish_perm(perms, i):
                    break               
                
                run = perms[i].pop(0)
                means[i].append(-data[i][run])
                nums[i] += 1
        minarm_index = np.argmin([np.mean(means[i]) for i in Arms])
        Arms.pop(minarm_index)
    
    return Arms[0], map(np.mean, means), nums    
        
        

def beta(s, budget, K, a, mean_list, automatic = False, auto_para = 1, H = 1):
    if automatic:
        alpha = auto_para

        a = alpha* (budget - K)/(4*H)
        return np.sqrt(a / s)
    else:
        return np.sqrt(a*(budget - K) / s)
            


def U(i, t, budget, pulls, a):
    K = len(pulls)
    return np.mean(pulls[i]) + beta(len(pulls[i]), budget, K, a)

def L(i, t, budget, pulls, a):
    K = len(pulls)
    return np.mean(pulls[i]) - beta(len(pulls[i]), budget, K, a)

def B(i, t, budget, pulls, a, K):
    list_woi = range(K)
    list_woi.pop(i)
    return np.max([U(j, t, budget, pulls, a) - L(i, t, budget, pulls, a)  for j in list_woi])


def calc_B(k, U_l, L_l, K, max_U_i_t_index, max_U_i_t, max_U_i_eq_k):
    if k == max_U_i_t_index:
        return max_U_i_eq_k - L_l[k]
    else:
        return max_U_i_t - L_l[k]


def UGapE(data, budget, a):
    K = len(data)    
    
    perms = [list(np.random.permutation(len(data[0]))) for i in range(K)]            
      
    #Initialization
    pulls = [[] for i in range(K)]
    nums = []
    for i in range(K):
        run = perms[i].pop(0)
        pulls[i].append(-data[i][run])
        nums.append(1)       
        
    #Exploration loop
    for round in range(K, budget):
        #Select an arm  
        mean_list = [np.mean(pulls[i]) for i in range(K)]
        beta_list = [beta(len(pulls[i]), budget, K, a, mean_list) for i in range(K)]


        U_list = [mean_list[i] + beta_list[i] for i in range(K)]
        L_list = [mean_list[i] - beta_list[i] for i in range(K)]


        max_U_i_t_index = np.argmax(U_list)
        max_U_i_t = U_list[max_U_i_t_index]
        max_U_i_eq_k = np.max(cp.copy(U_list).pop(max_U_i_t_index))

        B_list = [calc_B(k, U_list, L_list, K, max_U_i_t_index, max_U_i_t, max_U_i_eq_k) for k in range(K)]

        
        J_t = np.argmin([B_list[i] if nums[i] < 20 else 10**10 for i in range(K)])
        list_woJt = range(K)
        list_woJt.pop(J_t)
        u_t = list_woJt[np.argmax([U_list[i] if nums[i] < 20 else -10**10 for i in list_woJt])]
        l_t = J_t
                       
        
        I_t = [l_t, u_t][np.argmax([beta_list[i] for i in [l_t, u_t]])]                

        
        run = perms[I_t].pop(0)
        nums[I_t] += 1
        pulls[I_t].append(-data[I_t][run])
        

    
    #Select final result based on Beta
    mean_list = [np.mean(pulls[i]) for i in range(K)]
    beta_list = [beta(len(pulls[i]), budget, K, a, mean_list) for i in range(K)]
    U_list = [mean_list[i] + beta_list[i] for i in range(K)]
    L_list = [mean_list[i] - beta_list[i] for i in range(K)]


    max_U_i_t_index = np.argmax(U_list)
    max_U_i_t = U_list[max_U_i_t_index]
    max_U_i_eq_k = np.max(cp.copy(U_list).pop(max_U_i_t_index))

    B_list = [calc_B(k, U_list, L_list, K, max_U_i_t_index, max_U_i_t, max_U_i_eq_k) for k in range(K)]
    
    return np.argmin(B_list), mean_list, nums   
    #return np.argmax(mean_list), mean_list, nums

def UGapE_adaptive(data, budget, auto_para = 1):
    K = len(data)
    perms = [list(np.random.permutation(len(data[0]))) for i in range(K)]
    blogK = blog(K)

    #Initialization                                                                                                                                                        
    pulls = [[] for i in range(K)]
    nums = []
    for i in range(K):
        run = perms[i].pop(0)
        pulls[i].append(-data[i][run])
        nums.append(1)

    
    #Exploration loop                                                                                                                                                      
    for round in range(K, budget):
        #Select an arm                                                                                                                                                     
        mean_list = [np.mean(pulls[i]) for i in range(K)]

        #Estimation of H
        ucb_delta_list = [mean_list[i] + np.sqrt(1/(2*nums[i]))  for i in range(K)]
        H = np.sum([1/ucb_delta_list[i]**2 for i in range(K)])
        beta_list = [beta(len(pulls[i]), budget, K, 1, mean_list, automatic = True, auto_para = auto_para, H = H) for i in range(K)]
        


        U_list = [mean_list[i] + beta_list[i] for i in range(K)]
        L_list = [mean_list[i] - beta_list[i] for i in range(K)]


        max_U_i_t_index = np.argmax(U_list)
        max_U_i_t = U_list[max_U_i_t_index]
        max_U_i_eq_k = np.max(cp.copy(U_list).pop(max_U_i_t_index))

        B_list = [calc_B(k, U_list, L_list, K, max_U_i_t_index, max_U_i_t, max_U_i_eq_k) for k in range(K)]


        J_t = np.argmin([B_list[i] if nums[i] < 20 else 10**10 for i in range(K)])
        list_woJt = range(K)
        list_woJt.pop(J_t)
        u_t = list_woJt[np.argmax([U_list[i] if nums[i] < 20 else -10**10 for i in list_woJt])]
        l_t = J_t

        I_t = [l_t, u_t][np.argmax([beta_list[i] for i in [l_t, u_t]])]


        run = perms[I_t].pop(0)
        nums[I_t] += 1
        pulls[I_t].append(-data[I_t][run])



    #Select final result based on Beta                                                                                                                                    
    mean_list = [np.mean(pulls[i]) for i in range(K)]
    beta_list = [beta(len(pulls[i]), budget, K, 1, mean_list, automatic = True, auto_para = auto_para) for i in range(K)]
    U_list = [mean_list[i] + beta_list[i] for i in range(K)]
    L_list = [mean_list[i] - beta_list[i] for i in range(K)]

    #B_list = [calc_B(i, U_list, L_list) for i in range(K)]                                                                                                                
    max_U_i_t_index = np.argmax(U_list)
    max_U_i_t = U_list[max_U_i_t_index]
    max_U_i_eq_k = np.max(cp.copy(U_list).pop(max_U_i_t_index))
    #print 'K', K, len(U_list)                                                                                                                                             
    B_list = [calc_B(k, U_list, L_list, K, max_U_i_t_index, max_U_i_t, max_U_i_eq_k) for k in range(K)]

    return np.argmin(B_list), mean_list, nums
