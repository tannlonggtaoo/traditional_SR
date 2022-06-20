# A very traditional super-resolution method.
#
# Reference: Farsiu S, Robinson D, Elad M, Milanfar P. Fast and robust
# multi-frame super-resolution. IEEE Transactions on Image
# Processing, vol.13,no.10,pp.1327-1344, October, 2004.
#
# Blickwinkel 2022.6

import itertools
from typing import Callable
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import shift
from scipy.ndimage import convolve

def read_image(filename:str):
    img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
    return img

def psnr(img1,img2,maxvalue):
    mse = np.mean((img1-img2)**2)
    return 10*np.log10(maxvalue**2 / mse)

def get_frames_from_vid4(foldername:str,r,num_frame:int):
    '''
    get 0~num_frame-1 frames from video dataset vid4.
    foldername = "calendar" "city" "foliage" or "walk".
    returns frames_HR,frames_LR.
    '''
    frames_HR = []
    frames_LR = []
    with open(file="./Vid4/" + foldername + ".txt",mode="r") as f:
        frame_names = f.readlines()
        for name in frame_names[:num_frame]:
            frame = read_image("./Vid4/" + name[:-1])
            frames_HR.append(frame)
            frames_LR.append(cv2.resize(frame,(0,0),fx=1/r,fy=1/r))
    return np.stack(frames_HR),np.stack(frames_LR)


def Gk(X:np.ndarray,Yk:np.ndarray,mv:np.ndarray,H_ker:np.ndarray,r:int):
    '''
    The gradient of Lp error.
    X : the optimizing HR image
    Y = one of the LR frames
    mv : the overall motion vector with shape (2,)
    H_ker : blur kernel
    r : downsampling rate (integer, >1)
    '''
    down_X = convolve(shift(X,-mv),H_ker)[::r,::r]
    upper_diff = np.zeros_like(X)
    upper_diff[::r,::r] = 2 * (down_X - Yk > 0) - 1
    HT_ker = np.flip(np.flip(H_ker,axis=0),axis=1)
    return shift(convolve(upper_diff,HT_ker),mv)

def Rml(X:np.ndarray,l:int,m:int):
    '''
    The gradient of regularization term.
    X : the optimizing HR image
    l, m : distance of horizontal / vertical shift
    '''
    signs = 2 * ((X - shift(X,(l,m))) > 0) - 1
    r_grad = signs - shift(signs,(-l,-m))
    return  r_grad

def totalgrad(X,Y_seq,mvs,lamda,P,alpha,r,w=None):
    '''
    The gradient of total loss.
    X : the optimizing HR image
    Y_seq = [Y1,...,YN] : the LR frames
    mvs = [mv1,...,mvN] : the corresponding motion vectors
    lambda : regularization coefficient
    P,alpha : parameter of BTV (see original paper)
    w : weights for each frame, equal to 1/len(Y_seq) if assigned None
    '''
    # H_ker = np.array([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]]) / 273
    if w is None:
        w = np.full((len(Y_seq),),1/len(Y_seq))
    H_ker = np.array([[1,1,1],[1,1,1],[1,1,1]]) / 9
    G = np.zeros_like(X,dtype=np.float32)
    R = np.zeros_like(X,dtype=np.float32)
    X = X.astype(np.float32)
    Y_seq = Y_seq.astype(np.float32)
    for i,Y in enumerate(Y_seq):
        G += w[i] * Gk(X,Y,mvs[i],H_ker,r)
    for l,m in itertools.product(range(0,P+1),range(-P,P+1)):
        if l + m >= 0:
            R += alpha**(l+m) * Rml(X,l,m)
    return G + lamda * R / len(Y_seq)

def mv_estim(Y_seq):
    '''
    Estimating camera motion (translation, between base img Y0 and each following img) by Lucas-Kanade method.
    '''
    lk_params = dict( winSize  = Y_seq[0].shape,
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    mvs = []
    for Y in Y_seq:
        mv,_,_ = cv2.calcOpticalFlowPyrLK(Y_seq[0],Y,prevPts=np.array([[[0.0,0.0]]]).astype(np.float32),nextPts=None,**lk_params)
        mvs.append(mv)
    return np.stack(mvs).reshape(-1,2)


def gd(X_init,Y_seq,mvs_LR,lamda,P,alpha,beta,max_iter,r,verbose,ground_truth):
    '''
    X_init : the initial state of iteration
    beta : learning rate
    '''
    X = X_init.astype(np.float32)
    mvs = mvs_LR * r
    grad = totalgrad(X,Y_seq,mvs,lamda,P,alpha,r)
    for iter in range(max_iter):
        X = X - beta * grad
        #grad = totalgrad(X,Y_seq,mvs,lamda,P,alpha,r)
        grad = totalgrad(X,Y_seq,mvs,lamda,P,alpha,r)
        if verbose:
            X_as_pic = np.clip(X,0,255).astype(np.uint8)
            print(f"{iter}-th iteration : psnr = {psnr(ground_truth,X_as_pic,255)}")

    return np.clip(X,0,255).astype(np.uint8)

def img_test():
    r = 2
    Y_seq_HR, frames = get_frames_from_vid4("calendar",r=r,num_frame = 5)
    cv2.imwrite("gt.jpg",Y_seq_HR[0])
    mvs = mv_estim(frames)
    X_init = cv2.resize(src=frames[0],dsize=(0,0),fx=r,fy=r)
    cv2.imwrite("resize.jpg",X_init)
    cv2.imwrite("resize_cubic.jpg",cv2.resize(src=frames[0],dsize=(0,0),fx=r,fy=r,interpolation=cv2.INTER_CUBIC))
    X = gd(X_init,frames,mvs,lamda=0.09,P=1,alpha=0.5,beta=5,max_iter=20,r=r,verbose=True,ground_truth=Y_seq_HR[0])
    cv2.imwrite("out.jpg",X)
    print(psnr(Y_seq_HR[0],X,255))
    print(psnr(Y_seq_HR[0],X_init,255))

if __name__ == "__main__":
    img_test()
    print("done")