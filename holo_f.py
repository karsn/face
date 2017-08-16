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
    if type(gray) != np.ndarray and type(gray) != tuple:
        print("gfilter(): type of gray is error-{}".format(type(gray)))
        return None
        
    if type(g[0]) != np.ndarray:
        print("gfilter(): type of g is error-{}".format(type(g[0])))
        return None
        
    gs = list()
    
    accum = np.zeros_like(gray)
    for kern in g:
        fimg = cv2.filter2D(gray, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
        gs.append(accum)
    return gs
    
#########################
#gray = [[0,1,2,3,4,5,6,7],[...],[...],[...],[...],[...],[...],[...]]      
#gs = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]  

def g_trans(gray, gs):
    if type(gs[0]) != np.ndarray:
        print("g_trans(): type of gs is error-{}".format(type(gs[0])))
        return None
    
    #if type(gray) == tuple:
    #    return gfilter(gray,gs)    
        
    g_tree = []

    if type(gray) == tuple or type(gray)==np.ndarray:
        A = gfilter(gray,gs)
        #print(A)
        #PS = [0, 0.866, -0.866, 0]
        #PC = [1, -0.5, -0.5, 1]
        #
        #AS = A[0]*PS[0]+A[1]*PS[1]+A[2]*PS[2]+A[3]*PS[3]
        #AC = A[0]*PC[0]+A[1]*PC[1]+A[2]*PC[2]+A[3]*PC[3]
        #A0 = np.power(np.power(AS,2)+np.power(AC,2),0.5)
        #AS = A[0+4]*PS[0]+A[1+4]*PS[1]+A[2+4]*PS[2]+A[3+4]*PS[3]
        #AC = A[0+4]*PC[0]+A[1+4]*PC[1]+A[2+4]*PC[2]+A[3+4]*PC[3]
        #A1 = np.power(np.power(AS,2)+np.power(AC,2),0.5)
        #AS = A[0+8]*PS[0]+A[1+8]*PS[1]+A[2+8]*PS[2]+A[3+8]*PS[3]
        #AC = A[0+8]*PC[0]+A[1+8]*PC[1]+A[2+8]*PC[2]+A[3+8]*PC[3]
        #A2 = np.power(np.power(AS,2)+np.power(AC,2),0.5)
        #AS = A[0+12]*PS[0]+A[1+12]*PS[1]+A[2+12]*PS[2]+A[3+12]*PS[3]
        #AC = A[0+12]*PC[0]+A[1+12]*PC[1]+A[2+12]*PC[2]+A[3+12]*PC[3]
        #A3 = np.power(np.power(AS,2)+np.power(AC,2),0.5)
        ##AS = A[0*4+0]*PS[0]+A[1*4+0]*PS[1]+A[2*4+0]*PS[2]+A[3*4+0]*PS[3]
        ##AC = A[0*4+0]*PC[0]+A[1*4+0]*PC[1]+A[2*4+0]*PC[2]+A[3*4+0]*PC[3]
        ##A4 = np.power(np.power(AS,2)+np.power(AC,2),0.5)
        ##AS = A[0*4+1]*PS[0]+A[1*4+1]*PS[1]+A[2*4+1]*PS[2]+A[3*4+1]*PS[3]
        ##AC = A[0*4+1]*PC[0]+A[1*4+1]*PC[1]+A[2*4+1]*PC[2]+A[3*4+1]*PC[3]
        ##A5 = np.power(np.power(AS,2)+np.power(AC,2),0.5)
        ##AS = A[0*4+2]*PS[0]+A[1*4+2]*PS[1]+A[2*4+2]*PS[2]+A[3*4+2]*PS[3]
        ##AC = A[0*4+2]*PC[0]+A[1*4+2]*PC[1]+A[2*4+2]*PC[2]+A[3*4+2]*PC[3]
        ##A6 = np.power(np.power(AS,2)+np.power(AC,2),0.5)
        ##AS = A[0*4+3]*PS[0]+A[1*4+3]*PS[1]+A[2*4+3]*PS[2]+A[3*4+3]*PS[3]
        ##AC = A[0*4+3]*PC[0]+A[1*4+3]*PC[1]+A[2*4+3]*PC[2]+A[3*4+3]*PC[3]
        ##A7 = np.power(np.power(AS,2)+np.power(AC,2),0.5)
        #
        #Th = np.ones_like(A0)
        #Th = Th*255
        #A0 = np.fmin(A0,Th)
        #A1 = np.fmin(A1,Th)
        #A2 = np.fmin(A2,Th)
        #A3 = np.fmin(A3,Th)
        ##A4 = np.fmin(A4,Th)
        ##A5 = np.fmin(A5,Th)
        ##A6 = np.fmin(A6,Th)
        ##A7 = np.fmin(A7,Th)
        #
        #A0 = np.uint8(A0)
        #A1 = np.uint8(A1)
        #A2 = np.uint8(A2)
        #A3 = np.uint8(A3)
        ##A4 = np.uint8(A4)
        ##A5 = np.uint8(A5)
        ##A6 = np.uint8(A6)
        ##A7 = np.uint8(A7)
        
        A0 = np.amax(A[0:3],axis=0)
        A1 = np.amax(A[4:7],axis=0)
        A2 = np.amax(A[8:11],axis=0)
        A3 = np.amax(A[12:15],axis=0)
            
        A4 = np.amax(A[0:(len(gs)-1):4],axis=0)
        A5 = np.amax(A[1:(len(gs)-1):4],axis=0)
        A6 = np.amax(A[2:(len(gs)-1):4],axis=0)
        A7 = np.amax(A[3:(len(gs)-1):4],axis=0)
        
        g_tree.extend([A0,A1,A2,A3,A4,A5,A6,A7])
        
    elif type(gray) == list:
        for gray_sub in gray:
            g_tree.append(g_trans(gray_sub,gs))
    else:
        print("g_trans(): type of gray is error-{}".format(type(gray)))
        
    return g_tree

#########################
def create_pool_kern():
    filters = []
    ksize = [3] # gabor尺度，6个
    lamda = np.pi/2.0 #波长

    for theta in np.arange(0, np.pi, np.pi / 4): #gabor方向，0°，45°，90°，135°，共四个
        for K in range(len(ksize)): 
            kern = cv2.getGaborKernel((max(ksize), max(ksize)), 1.0, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5*kern.sum()
            filters.append(kern)      
            
    return filters   
#########################
# g_tree = [[,],[,]]
g_pool_kern = create_pool_kern()

def pool_max(g_tree,idx=0):
    global g_pool_kern
    
    res = []
    S = 0
    
    if type(g_tree) == list:
        for k,g_tree_sub in enumerate(g_tree):
            (A,S) = pool_max(g_tree_sub,k)
            res.append(A)
    elif type(g_tree) == np.ndarray:
        
        w = g_tree.shape[1]
        h = g_tree.shape[0]

        #if w<=5 or h<=5:
        A = np.zeros((int((h-3)/2+1),int((w-3)/2+1)),dtype=np.uint8)
        
        gx = 0
        if idx<4:
            gx = np.array([[0.25,0.5,0.25],[0.5,1,0.5],[0.25,0.5,0.25]])
            gx /= gx.sum()
        else:
            gx = g_pool_kern[idx-4]
            
        B = gfilter(g_tree, gx)
        
            
        for i in range(1,h-1,2):
            for k in range(1,w-1,2):
                A[int((i-1)/2)][int((k-1)/2)] = B[0][i][k] #np.amax(g_tree[0][0][(i-1):(i+1),(k-1):(k+1)])
                #A11[int((i-1)/2)][int((k-1)/2)] = np.amax(g_tree[0][0][(i-1):(i+1),(k-1):(k+1)])
                #A11[int((i-1)/2)][int((k-1)/2)] = g_tree[0][0][(i-1):(i+1),(k-1):(k+1)].sum()/9

        #print("pool_size=[{},{}]".format(A11.shape[0],A11.shape[1]))
        
        res = A
        S = min([res.shape[1], res.shape[0]])    
    else:
        print("pool_max(): g_tree's type is error-{}".format(type(g_tree)))   
                 
    return (res,S)

#########################
# g_tree = [[,],[,]]
def abstract(g_tree):
    res = []
    S = 0
    
    if type(g_tree) == list:
        for g_tree_sub in g_tree:
            (A,S) = abstract(g_tree_sub)
            res.append(A)
    elif type(g_tree) == np.ndarray:
        
        w = g_tree.shape[1]
        h = g_tree.shape[0]
        
        S = min([w,h])
        if S<=11:
            A = g_tree[int(h/2)][int(w/2)]
            S = 1
        else:
            print("abstract(): size error-[{},{}]".format(w,h))
            A = gtree
        
        #print("pool_size=[{},{}]".format(A11.shape[0],A11.shape[1]))
        
        res = A
            
    else:
        print("abstract(): g_tree's type is error-{}".format(type(g_tree)))   
                 
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
#print("1st conv")
#print(conv)
print("1")

# 2
pool,s = pool_max(conv)
#print("2nd pool")
#print(pool)
del conv
conv = g_trans(pool,gs)
del pool
print("2")

# 3
pool,s = pool_max(conv)
del conv
conv = g_trans(pool,gs)
del pool
print("3")

#4
pool,s = pool_max(conv)
del conv
conv = g_trans(pool,gs)
del pool
print("4")


##5
#pool,s = pool_max(conv)
#del conv
#conv = g_trans(pool,gs)
#del pool
#print("5")
#
##6
#pool,s = pool_max(conv)
#del conv
#conv = g_trans(pool,gs)
#del pool
#print("6")


disp,s = abstract(conv)
print("Rst")
print(disp)
#plt.imshow(disp)

#plt.show()
