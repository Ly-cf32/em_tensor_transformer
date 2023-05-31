#!/usr/bin/python

import numpy as np
import math
import time
import argparse
import pandas as pd
import scipy.sparse as sp

from scipy import float32
from scipy.stats import norm
from math import pow, log, log10
from scipy.sparse import coo_matrix



tr_len_eff = 1 #修改了一下
step_size = 0.85
frag_len_mean = 200
frag_len_std = 80
frag_len_max = 800
prior_intens = 10

# 得出pos_id和pos_num即区域长度的信息
def pos_generate(chr_id, start_pos, end_pos):
    pos_num = (end_pos - start_pos + 1)
    pos_id = []
    for i in range(start_pos, end_pos+1):
        pos_id.append(i)
    
    return pos_num, pos_id

# 生成01数组
def coocsr_generate(inarray, pos_num):
    value_sum = 0
    for i in range(pos_num):
        value_sum += inarray[i]
    
    num = int(value_sum)
    
    row_indices = []
    col_indices = []
    data = []

    v = 0
    for i, ind in enumerate(inarray):
        row_indices.extend(range(v, v + ind))
        col_indices.extend([i] * ind)
        data.extend([1] * ind)
        v += ind

    # 根据行、列和数据数组创建稀疏矩阵
    b = coo_matrix((data, (row_indices, col_indices)), shape=(num, inarray.shape[0]))
    b_csr = b.tocsr()
    
    return b_csr


def main():
    usage = """usage: %(prog)s [options] <-p pfile> <-m mfile>  
Example: %(prog)s -p pure.bw -m mix.bw -P 0.9 -l 100 -a type_a -b type_b -A 0 
"""
    parser = argparse.ArgumentParser(usage = usage)
    
    parser.add_argument("-p", "--pure-sample", dest="inw_p", type=str, help="position coverage file of the pure sample in npy format. REQUIRED")
    parser.add_argument("-m", "--mixed-sample", dest="inw_m", type=str, help="position coverage file of the mixed sample in npy format. REQUIRED")
    parser.add_argument("-a", "--type-a", dest="outreads_a", type=str, help="the name of the output file of cell type a, which is the only cell type of the pure sample. DEFAULT: \"type_a\"", default='type_a')
    parser.add_argument("-b", "--type-b", dest="outreads_b", type=str, help="the name of the output file of cell type b, which is the second cell type within the mixed sample. DEFAULT: \"type_b\"", default='type_b')
    parser.add_argument("-P", "--type-b-proportion", dest="b_prop", type=str, help="cell type b proportion. e.g. \"-P 0.9\". REQUIRED")
    parser.add_argument("-A", "--additional-rounds", dest="add_rounds", type=str, help="the number of addtional rounds of EM algorithm after the first online round. DEFAULT: 0", default='0')    
    
    args = parser.parse_args()
    
    err_flag = False
    if args.inw_p == None:
        print("Error: No read alignment file of pure sample!")
        err_flag = True
        
    if args.inw_m == None:
        print("Error: No read alignment file of mixed sample!")
        err_flag = True
    
    if args.b_prop == None:
        print("Error: No cell type b proportion!")
        err_flag = True
    
    if args.b_prop == None:
        print("Error: No read length!")
        err_flag = True
    
    if err_flag == True:
        print('\n')
        parser.print_help()
        parser.exit()
        
    inw_p = np.load(args.inw_p)
    filename_p = args.inw_p.split('/')[len(args.inw_p.split('/')) - 1]
    inw_m = np.load(args.inw_m)
    filename_m = args.inw_m.split('/')[len(args.inw_m.split('/')) - 1]
    outreads_a = open(args.outreads_a+'.temt', 'w')
    outreads_xa = open(args.outreads_a+'_xa.temt', 'w') # 
    outreads_b = open(args.outreads_b+'.temt', 'w')
    wa = 1-float(args.b_prop)
    wb = float(args.b_prop)
    add_rounds = int(args.add_rounds)

    time_start = time.time()

    
####Online EM step###########################
    
    pos_num, pos_id = pos_generate('chr14', 105854220, 105856218) # 得到位点信息
    
    print('generating array_a and array_m...')
    array_a_pri = coocsr_generate(inw_p, pos_num)
    array_m_pri = coocsr_generate(inw_m, pos_num)
    print('array_a and array_m generated.')
    
    alpha_a = np.array(np.ones(pos_num)/pos_num, dtype=float32) #initial probabilites of chosing 1 read from transcripts of normal cells
    alpha_b = np.array(np.ones(pos_num)/pos_num, dtype=float32) #initial probabilites of chosing 1 read from transcripts of tumor cells
    q_a = np.array(np.ones(pos_num)/pos_num, dtype=float32) #initial EM weight for each transcript of read_i from pure normal cells
    q_xa = np.array(np.ones(pos_num)/(2*pos_num), dtype=float32) #initial EM weight for each transcript of read_i from normal cells within mixture
    q_xb = np.array(np.ones(pos_num)/(2*pos_num), dtype=float32) #initial EM weight for each transcript of read_i from tumor cells within mixture
    tau_a = 0.5 #np.random.rand() #proportion of normal cells within mixture
    tau_b = 1 - tau_a #proportion of tumor cells within mixture
    i_a = 0
    i_x = 0    
    
    
    print('processing reads file...')
    for j in range(0, add_rounds+1):
        read_num_a = 0
        read_num_x = 0
        
        array_a = array_a_pri
        array_m = array_m_pri

        rows = max(array_a.shape[0], array_m.shape[0])
        
        current_time = time.time()
        for i in range(rows):
            #
            #Online E-step
            if i < array_a.shape[0]:
                Y_a_i = array_a.getrow(i).toarray()[0]

                likelihood_a_i = Y_a_i*alpha_a/tr_len_eff # bias 模块修改
                q_a_i = likelihood_a_i/np.sum(likelihood_a_i)
                q_a = (1-1/pow(i_a+2, step_size))*q_a + (1/pow(i_a+2, step_size))*q_a_i

                #Online M-step
                alpha_a = (q_a + q_xa)/(1 + tau_a)

                i_a = i_a + 1
                read_num_a = read_num_a + 1

                if read_num_a%100000 == 0:
                    run_time = time.time() - current_time
                    current_time = time.time()
                    print('Round %s\t%s position counts in %s processed...' % (j+1, read_num_a, filename_p)) # filename_p
            else:
                pass


            if i < array_m.shape[0]:
                #Online E-step
                Y_x_i = array_m.getrow(i).toarray()[0]
                likelihood_xa_i = Y_x_i*alpha_a/tr_len_eff
                likelihood_xb_i = Y_x_i*alpha_b/tr_len_eff # bias 模块修改

                q_xa_i = likelihood_xa_i*tau_a/np.sum(likelihood_xa_i*tau_a + likelihood_xb_i*tau_b)
                q_xb_i = likelihood_xb_i*tau_b/np.sum(likelihood_xa_i*tau_a + likelihood_xb_i*tau_b)
                q_xa = (1-1/pow(i_x+2, step_size))*q_xa + (1/pow(i_x+2, step_size))*q_xa_i
                q_xb = (1-1/pow(i_x+2, step_size))*q_xb + (1/pow(i_x+2, step_size))*q_xb_i                

                #Online M-step
                tau_a = (np.sum(q_xa) + wa*prior_intens)/(1 + prior_intens) # wa,wb,prior_intens
                tau_b = (np.sum(q_xb) + wb*prior_intens)/(1 + prior_intens)
                alpha_a = (q_a + q_xa)/(1 + tau_a)
                alpha_b = q_xb/tau_b

                i_x = i_x + 1
                read_num_x = read_num_x + 1
                
                if read_num_x%100000 == 0:
                    run_time = time.time() - current_time
                    current_time = time.time()
                    print('Round %s\t%s position counts in %s processed...' % (j+1, read_num_x, filename_m)) # filename_m
            else:
                pass
        
        
############finalizing  Round####################

    read_num_a = 0
    read_num_x = 0
    est_counts_a = np.array(np.zeros(pos_num), dtype=float32)
    est_counts_xa = np.array(np.zeros(pos_num), dtype=float32)
    est_counts_xb = np.array(np.zeros(pos_num), dtype=float32)

    current_time = time.time()
    for i in range(array_a.shape[0]):
        Y_a_i = array_a.getrow(i).toarray()[0]
        
        likelihood_a_i = Y_a_i*alpha_a/tr_len_eff #bias模块修改
        q_a_i = likelihood_a_i/np.sum(likelihood_a_i)
        est_counts_a = est_counts_a + q_a_i #estimated counts for each transcripts of pure normal cells based on reads_a
        read_num_a = read_num_a + 1
        if read_num_a%100000 == 0:
            run_time = time.time() - current_time
            current_time = time.time()
            print('Finalizing %s\t%s position counts processed...' % (filename_p, read_num_a))
        
    for i in range(array_m.shape[0]):
        Y_x_i = array_m.getrow(i).toarray()[0]
        
       
        likelihood_xa_i = Y_x_i*alpha_a/tr_len_eff
        likelihood_xb_i = Y_x_i*alpha_b/tr_len_eff # bias模块修改
        # 增多est_count_a
        q_xa_i = likelihood_xa_i*tau_a/np.sum(likelihood_xa_i*tau_a + likelihood_xb_i*tau_b) #
        q_xb_i = likelihood_xb_i*tau_b/np.sum(likelihood_xa_i*tau_a + likelihood_xb_i*tau_b)
        est_counts_xa = est_counts_xa + q_xa_i #
        est_counts_xb = est_counts_xb + q_xb_i #esimated counts for each transcripts of tumor cells within the mixture based on reads_x
        read_num_x = read_num_x + 1
        if read_num_x%100000 == 0:
            run_time = time.time() - current_time
            current_time = time.time()
            print('Finalizing %s\t%s position counts processed...' % (filename_m, read_num_x))
        
    read_num_xa = np.sum(est_counts_xa)
    read_num_xb = np.sum(est_counts_xb)
    
###############Finishing##################

    time_end = time.time()
    print('Time used: %ssec' % (time_end-time_start))

    outreads_a.write('pos_ID\testimated_counts\tRPKM\n')
    for i in range(0, pos_num):
        outreads_a.write('%s\t%s\t%s\n' % (pos_id[i], est_counts_a[i], est_counts_a[i]*(10**9)/(read_num_a*100))) # 这儿100原为tr_len[i]
    
    # 增多xa的写入
    outreads_xa.write('pos_ID\testimated_counts\tRPKM\n')
    for i in range(0, pos_num):
        outreads_xa.write('%s\t%s\t%s\n' % (pos_id[i], est_counts_xa[i], est_counts_xa[i]*(10**9)/(read_num_xb*100))) # 这儿100原为tr_len[i]
        
    outreads_b.write('pos_ID\testimated_counts\tRPKM\n')  
    for i in range(0, pos_num):
        outreads_b.write('%s\t%s\t%s\n' % (pos_id[i], est_counts_xb[i], est_counts_xb[i]*(10**9)/(read_num_xb*100))) # 这儿100原为tr_len[i]
    
    # inw_p.close()
    # inw_m.close()
    outreads_a.close()
    outreads_xa.close() #
    outreads_b.close()
    
if __name__ == "__main__":
    main()
