#/usr/bin/env python2
import scipy,numpy,math,sys
from scipy.io.wavfile import read
from scipy.cluster.vq import kmeans2, kmeans, vq
import soundfile as sf
from scipy.signal import hann, lfilter, freqz, decimate, butter
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite
from scipy.signal import hann, lfilter, freqz, decimate, butter, medfilt
from numpy import array, double, amax, absolute, zeros, floor, arange, mean
from numpy import correlate, dot, append, divide, argmax, int16, sqrt, power,corrcoef,std
from numpy.random import randn
from scikits.talkbox.linpred import lpc
import matplotlib
import matplotlib.pyplot as plt
from numpy import array, double, amax, absolute, zeros, floor, arange, mean

FRAME_SIZE=160
OVERLAP=0
def load_file(fn):
    out=[]
    with open(fn,'rb') as infile:
        data = numpy.fromfile(fn, dtype=numpy.uint8)
        #data=infile.read()#.split(" ")
        for idx in xrange(0,len(data)-3,3): #10*lpc+e+f0
            out.append(int(data[idx]))
            out.append(int(data[idx+1]))
            out.append(int(data[idx+2]))

    return (idx+3)/3,out # pocet vzorku a vystupni parametry


def decode(samples,enc,codebook_lpc,codebook_e):
    #b, aa = butter(1, 0.01, 'low')

    #flt_f0=medfilt(enc[2::3],kernel_size=5)
    #print(enc[2::3])
    out=zeros(samples*(FRAME_SIZE-OVERLAP))
    for idx in xrange(0,len(enc),3):
        a_idx=enc[idx]
        e_idx=enc[idx+1]
        #f0_sam=int(flt_f0[idx/3])#
        f0_sam =enc[idx+2]
        print (f0_sam)
        if f0_sam !=0:
            vocoded = zeros(FRAME_SIZE)
            vocoded[f0_sam::f0_sam] = 1.0
        else:
            vocoded = numpy.random.random(FRAME_SIZE)/4#???
        #vocoded -= lfilter(b, aa, vocoded)

        e=codebook_e[e_idx]
        a=codebook_lpc[a_idx]
        vocoded = lfilter((e,), a, vocoded)

        try:
            out[idx/3 * (FRAME_SIZE-OVERLAP):idx/3 * (FRAME_SIZE-OVERLAP) + FRAME_SIZE] += vocoded
        except Exception as e:
            print (e)
    return out

def load_codebook(fn):
    return numpy.load(fn)
codebook_lpc=load_codebook("codebook_lpc.npy")
codebook_e=load_codebook("codebook_gain.npy")
idx,data=load_file(sys.argv[1])
out=decode(idx,data,codebook_lpc,codebook_e)
wavwrite(sys.argv[2], 8000, array(out/amax(absolute(out)) * (2**15-1), dtype=int16))
