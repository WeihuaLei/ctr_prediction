#!/sur/bin/env python

import math
import random
import sys,datetime

#train_feature
train_file = open(sys.argv[1], "r")

#when we proform prediction, the validate_file is "test_feature"
validate_file = open(sys.argv[2], "r")


alpha = 0.40
beta = 1
l1 = 0.0
l2 = 0.05

#this is the max feature index of train and test data
max_index = 6583357
"""
for f in test_file:
    seg = f.strip().split("\t")
    for st in seg[1:]:
        index = int(st.split(":")[0])
        if index > max_index:
        max_index = index
file.close()
"""


z = [0]*(max_index+1)
n = [0]*(max_index+1)
weights = [0]*(max_index+1)

n1 = datetime.datetime.now()    

for f in train_file:
        seg = f.strip().split("\t")
        label = int(seg[0])
        #compute w_t,i
        for st in seg[1:]:
            index_val = st.strip().split(":")
            index = int(index_val[0])
            val = float(index_val[1])
            if abs(z[index]) <= l1:
                weights[index] = 0
            elif z[index] > l1:
                weights[index] = -1.0 / ((beta + math.sqrt(n[index]))/alpha + l2) * (z[index] - l1)
            else:
                weights[index] = -1.0 / ((beta + math.sqrt(n[index]))/alpha + l2) * (z[index] + l1)
        #predict x_t    
        s = 0.0
        for st in seg[1:]:
            index_val = st.strip().split(":")
            index = int(index_val[0])
            val = float(index_val[1])
            s += weights[index]*val
        
        p = 1.0/(1 + math.exp(-s))
        g = p - label
        
        #update z_i, n_i
        for st in seg[1:]:
            index_val = st.strip().split(":")
            index = int(index_val[0])
            val = float(index_val[1])
            
            g_i = g * val
            sigma_i = 1.0/alpha * (math.sqrt(n[index] + g_i**2) - math.sqrt(n[index]))
            z[index] += (g_i - sigma_i*weights[index])
            n[index] += g_i**2

non_zero = 0
for w in weights:
    if w!=0:
        non_zero += 1

#wirte the prediction of test data
to_write = open("predict_ftrl", "w+")

#when perform the prediction in test_feature,the column of "label" 
#is the line number of test_feature. we will collect the prediction only

for f in validate_file:
    seg = f.strip().split("\t")
    label = int(seg[0])
    s = 0.0
    for st in seg[1:]:
        index_val = st.split(":")
        index = int(index_val[0])
        val = float(index_val[1])
        s += weights[index]*val
    p = 1.0 / (1 + math.exp(-s))
    label_pred = seg[0] + "," + str(p) + "\n"
    to_write.write(label_pred)

to_write.close()

n2 = datetime.datetime.now()

# write some logging
logfile = open("logging", "a+")

now = datetime.datetime.now()
logfile.write("\n******This is a new try*****\n")
logfile.write("Current time:" + str(now) + "\n")
params = "alpha is:{0} \t l1 is:{1} \t l2 is:{2} \n".format(alpha,l1,l2)
logfile.write(params)
logfile.write("Length of non-zero weights is:%d \n"%non_zero)
logfile.write("this try costs time: %s \n"%(n2-n1))
logfile.close()

