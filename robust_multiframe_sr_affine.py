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
    size_HR = (X.shape[1],X.shape[0])
    # test
    # global glb_k
    # cv2.imwrite(f"test{glb_k}.jpg",cv2.warpAffine(X,mv,dsize=size_HR,flags=cv2.WARP_INVERSE_MAP))
    # glb_k+=1
    # end of test

    down_X = convolve(cv2.warpAffine(X,mv,dsize=size_HR),H_ker)[::r,::r]
    upper_diff = np.zeros_like(X)
    upper_diff[::r,::r] = 2 * (down_X - Yk > 0) - 1
    HT_ker = np.flip(np.flip(H_ker,axis=0),axis=1)
    return cv2.warpAffine(convolve(upper_diff,HT_ker),mv,dsize=size_HR)

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

def getAffine_LS(src,dst):
    '''
    compute least-square Affine mat by pts from src to dst
    '''
    src = src.reshape(-1,2)
    X = np.c_[src,np.full((len(src),1),1)]
    dst = dst.reshape(-1,2)
    invpart = (np.linalg.inv(np.matmul(X.transpose(),X)))
    A = np.matmul(np.matmul(invpart,X.transpose()),dst)
    return A.transpose()

def mv_estim(Y_seq,num_corner):
    '''
    Estimating global motion (translation & scaling, between base img Y0 and each following img) by Lucas-Kanade method.
    '''
    corners = cv2.goodFeaturesToTrack(Y_seq[0],maxCorners=num_corner,qualityLevel=0.01,minDistance=20)
    better_corners = cv2.cornerSubPix(Y_seq[0],corners,(10,10),(-1,-1),(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,20,0.03))
    w, h = Y_seq[0].shape
    lk_params = dict( winSize  = (w,h),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    mvs = []
    for Y in Y_seq:
        nextPts,_,_ = cv2.calcOpticalFlowPyrLK(Y_seq[0],Y,prevPts=better_corners,nextPts=None,**lk_params)
        mv = getAffine_LS(corners,nextPts)
        mvs.append(mv)
    return np.stack(mvs)


def gd(X_init,Y_seq,mvs_LR,lamda,P,alpha,beta,max_iter,r,verbose,ground_truth):
    '''
    X_init : the initial state of iteration
    beta : learning rate
    '''
    X = X_init.astype(np.float32)
    mvs = np.c_[mvs_LR[:,:,:-1],mvs_LR[:,:,-1].reshape(-1,2,1) * r]
    weight = np.exp(-np.linalg.norm(mvs[:,:,-1],axis=1))
    weight = weight / np.sum(weight)
    grad = totalgrad(X,Y_seq,mvs,lamda,P,alpha,r)
    for iter in range(max_iter):
        X = X - beta * grad
        grad = totalgrad(X,Y_seq,mvs,lamda,P,alpha,r,weight)
        if verbose:
            X_as_pic = np.clip(X,0,255).astype(np.uint8)
            print(f"{iter}-th iteration : psnr = {psnr(ground_truth,X_as_pic,255)}")
            cv2.imwrite("out.jpg",X)

    return np.clip(X,0,255).astype(np.uint8)

def img_test():
    r = 2
    Y_seq_HR, frames = get_frames_from_vid4("calendar",r=r,num_frame = 10)
    cv2.imwrite("gt.jpg",Y_seq_HR[0])
    mvs = mv_estim(frames,100)
    X_init = cv2.resize(src=frames[0],dsize=(0,0),fx=r,fy=r)
    
    cv2.imwrite("resize.jpg",X_init)
    cv2.imwrite("resize_cubic.jpg",cv2.resize(src=frames[0],dsize=(0,0),fx=r,fy=r,interpolation=cv2.INTER_CUBIC))
    X = gd(X_init,frames,mvs,lamda=0.3,P=2,alpha=0.6,beta=2,max_iter=20,r=r,verbose=True,ground_truth=Y_seq_HR[0])
    cv2.imwrite("out.jpg",X)
    print(psnr(Y_seq_HR[0],X,255))
    print(psnr(Y_seq_HR[0],X_init,255))

if __name__ == "__main__":
    img_test()
    print("done")