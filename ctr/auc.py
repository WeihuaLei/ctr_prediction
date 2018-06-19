#!/usr/bin/env python
# -*- coding:utf-8 -*-
from sklearn.metrics import auc,roc_curve

import numpy as np
"""
validate the model on validation data

"""

file = open("pctr","r")

label = []
p = []

for f in file:
   seg = f.strip().split(",")
   label.append(int(seg[0]))
   p.append(float(seg[1]))

y = np.array(label)
pred = np.array(p)
a, b, c = roc_curve(y, pred ,pos_label = 1)

auc = auc(a, b)
print auc
logfile = open("logging", "a+")
logfile.write("the auc of current trail is %f \n" % auc)
logfile.write("****this try ends*****\n\n\n")
