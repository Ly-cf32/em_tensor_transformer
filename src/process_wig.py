#!/usr/bin/python
import numpy as np
import pandas as pd
import sys

# generate the coverage of every position and normalise to 100W reads

# smooth the data
def smooth_wig_gen(input_wig, output_wig, output_smoothed_wig, length, read_num):
    with open(input_wig,'r') as inwig:
        #open(output_wig, 'w') as outwig, open(output_smoothed_wig,'w') as outsmwig
        try:
            wig = pd.read_csv(inwig,sep = '\t')
            
            np_wig = np.zeros(length)

            # 转化为每个位点的值
            #for i,ind in enumerate(wig.index-105736343):
            for i,ind in enumerate(wig.index-105854220): # for ighm
                np_wig[ind]=wig['variableStep chrom=chr14'].iloc[i]
            np.save(output_wig,np_wig)
            #np.savetxt(output_wig,np_wig)


            # 均一化成100W reads
            np_wig_gen = np_wig/read_num * 1e6

            # 平滑化
            window_size = 50 # 设置窗口大小，可以根据需要调整 
            smoothed_wig = np.convolve(np_wig_gen, np.ones(window_size),'same') / window_size
            smoothed_wig = smoothed_wig.astype(int)
            np.save(output_smoothed_wig,smoothed_wig)
            #np.savetxt(output_smoothed_wig,smoothed_wig)
        except Exception as e:
            #print(e, type(e))
            if(isinstance(e, pd.errors.EmptyDataError)):
                print('Ignoring the empty files...')
           

    

if __name__ == "__main__":
    input_wig = sys.argv[1]
    output_wig = sys.argv[2]
    output_smoothed_wig = sys.argv[3]
    read_num = int(sys.argv[4])
    
    length = 105856218-105854220 # for ighm
    
    smooth_wig_gen(input_wig, output_wig, output_smoothed_wig, length, read_num)
