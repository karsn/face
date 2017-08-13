#!/usr/bin/python
#coding=utf-8

import os
import cv2
import time
import sys
import numpy as np
import matplotlib.pyplot as plt

#########################
# g_tree = [[,],[,]]
# res = 2D array
def merger(g_tree):
    if type(g_tree[0][0]) == list:
        A11 = merger(g_tree[0][0])
        A12 = merger(g_tree[0][1])
        A21 = merger(g_tree[1][0])
        A22 = merger(g_tree[1][1])
        
        #res = np.array([[A11,A12],[A21,A22]])
        res = np.vstack((np.hstack((A11,A12)),np.hstack((A21,A22))))
    else:
        #res = np.array(g_tree)
        res = np.vstack((np.hstack((g_tree[0][0],g_tree[0][1])),np.hstack((g_tree[1][0],g_tree[1][1]))))
    return res


#########################
def g_create():
    filters = []
    ksize = [5,7,9,11] # gabor尺度，6个
    lamda = np.pi/2.0 #波长

    for theta in np.arange(0, np.pi, np.pi / 4): #gabor方向，0°，45°，90°，135°，共四个
        for K in range(len(ksize)): 
            kern = cv2.getGaborKernel((max(ksize), max(ksize)), 1.0, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5*kern.sum()
            filters.append(kern)      
            
    return filters
    
#########################
#return:[g0,g1,...,g15,...]
def gfilter(gray, g):
    if type(gray) != np.array:
        print("gfilter(): type of gray is error-{}".format(type(gray)))
        return None
        
    if type(g[0]) != np.array:
        print("gfilter(): type of g is error-{}".format(type(g[0])))
        return None
        
    gs = list()
    
    accum = np.zeros_like(gray)
    for kern in filters:
        fimg = cv2.filter2D(gray, cv2.CV_8UC3, gs)
        np.maximum(accum, fimg, accum)
        gs.append(accum)
    return gs
    
#########################
#gray = [[0,1,2,3,4,5,6,7],[...],[...],[...],[...],[...],[...],[...]]      
#gs = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]  

def g_trans(gray, gs):
    if type(gs[0]) != np.array:
        print("g_trans(): type of gs is error-{}".format(type(g[0])))
        return None
    
    if type(gray) == np.array:
        return gfilter(gray,gs)    
        
    g_tree = []

    if type(gray) == np.array:
        A = gfilter(gray,gs)
        A0 = max(A[0:3])
        A1 = max(A[4:7])
        A2 = max(A[8:11])
        A3 = max(A[12:15])
            
        A4 = max(A[0:(len(gs)-1):4])
        A5 = max(A[1:(len(gs)-1):4])
        A6 = max(A[2:(len(gs)-1):4])
        A7 = max(A[3:(len(gs)-1):4])
        
        g_tree.extend([A0,A1,A2,A3,A4,A5,A6,A7])
        
    elif type(gray) == list:
        for gray_sub in gray:
            g_tree.append(g_trans(gray_sub,gs))
    else:
        print("g_trans(): type of gray is error-{}".format(type(gray)))
        
    return g_tree
   
#########################
# g_tree = [[,],[,]]
def pool_max(g_tree):
    res = []
    S = 0
    
    if type(g_tree) == list:
        for g_tree_sub in g_tree:
            (A,S) = pool_max(g_tree_sub)
            res.append(A)
    elif type(g_tree) == np.array:
        
        w = g_tree.shape[1]
        h = g_tree.shape[0]

        #if w<=5 or h<=5:
        A = np.zeros((int((h-3)/2+1),int((w-3)/2+1)),dtype=np.uint8)
        
        gx = np.array([[0.25,0.5,0.25],[0.5,1,0.5],[0.25,0.5,0.25]])
        gx /= gx.sum()
        
        B = gfilter(g_tree, gx)
        
            
        for i in range(1,h-1,2):
            for k in range(1,w-1,2):
                A[int((i-1)/2)][int((k-1)/2)] = B11[i][k] #np.amax(g_tree[0][0][(i-1):(i+1),(k-1):(k+1)])
                #A11[int((i-1)/2)][int((k-1)/2)] = np.amax(g_tree[0][0][(i-1):(i+1),(k-1):(k+1)])
                #A11[int((i-1)/2)][int((k-1)/2)] = g_tree[0][0][(i-1):(i+1),(k-1):(k+1)].sum()/9

        #print("pool_size=[{},{}]".format(A11.shape[0],A11.shape[1]))
        
        res = A
        S = min([w,h])    
    else:
        print("pool_max(): g_tree's type is error-{}".format(type(g_tree)))   
                 
    return (res,S)
        
if len(sys.argv) != 2:
    print("input img")
    exit()  
    
img_path = sys.argv[1] #"."      
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgW = img.shape[1]
imgH = img.shape[0]

stride = 2
np.set_printoptions(threshold = 1e6)

gs = g_create()

# 1
conv = g_trans(gray,gs)

# 2
pool = pool_max(conv)
del conv
conv = g_trans(pool,gs)
del pool
print("2")

# 3
pool = pool_max(conv)
del conv
conv = g_trans(pool,gs)
del pool
print("3")

#4
pool = pool_max(conv)
del conv
conv = g_trans(pool,gs)
del pool
print("4")

#
##5
#pool = pool_max(conv)
#del conv
#conv = g_trans(pool,gs)
#del pool
#print("5")
#
##6
#pool = pool_max(conv)
#del conv
#conv = g_trans(pool,gs)
#del pool
#print("6")


disp = merger(conv)
print(disp)
#plt.imshow(disp)

plt.show()
