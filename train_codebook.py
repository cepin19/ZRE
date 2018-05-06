#/usr/bin/env python2

from scipy.signal import hann, lfilter, freqz, decimate, butter
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite
#from scikits.talkbox.linpred import lpc
import numpy
import os
import math
from scipy.cluster.vq import kmeans2, kmeans, vq
from numpy import array, double, amax, absolute, zeros, floor, arange, mean
from numpy import correlate, dot, append, divide, argmax, int16, sqrt, power,corrcoef,std

LPC=10
FRAME_SIZE=160
OVERLAP=0


#levinson algorithm by https://github.com/bastibe/pocoder
def levinson_algorithm(r):
    a = zeros(len(r));
    k = zeros(len(r));
    for m in arange(0,len(r)-1):
        alpha = -dot( r[m::-1], append(a[m:0:-1], 1.0) )
        mu = -dot(r[m::-1], append(a[1:m+1], 0.0)) - r[m+1]
        k[m] = -mu / alpha
        a[1:m+2] = append(a[1:m+1], 0.0) + k[m] * append(a[m:0:-1], 1.0)

    a[0] = 1
    return (a,k)
def rms(signal):
    return sqrt(mean(power(signal, 2)))
def preemphasis(signal):
    return lfilter([1, -0.70], 1, signal)

def deemphasis(signal):
    return lfilter([1, 0.70], 1, signal)



def make_frames(signal):
    num_frames=(len(signal)/(FRAME_SIZE-OVERLAP))
    frames=numpy.ndarray((num_frames,FRAME_SIZE)) # This declares a 2D matrix,with rows equal to the number of frames,and columns equal to the framesize or the length of each DTF
    for k in range(0,num_frames):
        for i in range(0,FRAME_SIZE):
            if((k*(FRAME_SIZE-OVERLAP)+i)<len(signal)):
                frames[k][i]=signal[k*(FRAME_SIZE-OVERLAP)+i]
            else:
                frames[k][i]=0
    return frames
def train_codebook(train_files):
    lpcs=[]
    es=[]
    for fn in train_files:
        signal=wavread("data/"+fn)[1]/2.0**15
        for idx, frame in enumerate(make_frames(signal)):

            frame -= numpy.mean(signal)
            #a, e, _ = lpc(frame, LPC)
            #e = sqrt(e)[0]
            framep = preemphasis(frame)
            rxx = correlate(framep, framep, mode='full')
            rxx = rxx[len(rxx) / 2:]
            a, k = levinson_algorithm(rxx[:LPC + 1])
            e = sqrt(mean(lfilter(a, (1,), framep) ** 2))
            lpcs.append(a)
            es.append(e)

    try:
        clusters_lpc = kmeans(lpcs, 256)[0]
        clusters_e = kmeans(es, 256)[0]
    except:
        clusters_lpc=kmeans(lpcs,idx-1)[0]
        clusters_e = kmeans(es, idx-1)[0]

    return clusters_lpc,clusters_e

if __name__=="__main__":
    files=os.listdir("data/")
    clusters_lpc, clusters_e=train_codebook(["test.wav"])
    numpy.save("codebook_lpc.npy",clusters_lpc)
    numpy.save("codebook_gain.npy",clusters_e)