#! /usr/bin/python

import random
import math
import sys
import datetime
import pickle

alpha = 0.15

iter = 1

l2 = 0.0


file = open("test_feature", "r")

n1 = datetime.datetime.now()

max_index = 0

for f in file:
     seg = f.strip().split("\t")
     for st in seg[1:]:
         index = int(st.split(":")[0])
         if index > max_index:
            max_index = index
file.close()


weight = range(max_index+1)
delta = [0] * (max_index+1)
for i in range(max_index+1):
    weight[i] = random.uniform(-0.05, 0.05)

for i in range(iter):
    file = open(sys.argv[1], "r")

    for f in file:
       seg = f.strip().split("\t")
       label = int(seg[0])
       s = 0.0
       for st in seg[1:]:
           index_val = st.split(":")
           index = int(index_val[0])
           val = float(index_val[1])
           s += weight[index]*val
       p = 1.0 / (1 + math.exp(-s))
       g = p - label
       for st in seg[1:]:
           index = int(st.split(":")[0])
           val = float(st.split(":")[1])
           #common SGD
           #weight[index] -=  alpha * (g * val + l2 * weight[index])
           
           #adaSGD
           delta[index] += (g*val)**2
           weight[index]  -= alpha/(1 + math.sqrt(delta[index]))*(g*val + l2*weight[index])

#with open("weights.pickle","w+") as w:
#    pickle.dump(weight, w)

#test data
file = open(sys.argv[2], "r")

to_write = open("pctr", "w+")

for f in file:
   seg = f.strip().split("\t")
   label = int(seg[0])
   s = 0.0
   for st in seg[1:]:
       index_val = st.split(":")
       index = int(index_val[0])
       val = float(index_val[1])
       s += weight[index]*val
   p = 1.0 / (1 + math.exp(-s))
   label_pred = seg[0] + "," + str(p) + "\n"
   to_write.write(label_pred)

to_write.close()
n2 = datetime.datetime.now()

logfile = open("logging", "a+")

now = datetime.datetime.now()
logfile.write("\n******This is a new try*****\n")
logfile.write("Current time:" + str(now) + "\n")
params = "alpha is:{0} \t iteration is:{1} \t regularized L is:{2} \n".format(alpha,iter,l2)
logfile.write(params)
logfile.write("Length of weights is:%d \n"%len(weight))
logfile.write("this try costs %s times\n"%(n2-n1))
logfile.close()


