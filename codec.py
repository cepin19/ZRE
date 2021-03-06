#/usr/bin/env python2

import scipy,numpy,math
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
LPC=10
numpy.set_printoptions(threshold=numpy.nan)

# tuhle fci potom muzeme pouzit na trenovani codebooku pro LPC gain
sample_rate,data=read("test.wav")
#print (sample_rate)
data=data.astype('float32')
#print (data)
clusters=kmeans2(data, 10)
#print (clusters)
#print (data.shape)

#TODO neukladat 0. lpc koef, zjistit, proc je kvalita tak spatna, nahravat raw soubory




fs=8000

"""

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

"""

def load_codebook(fn):
    return numpy.load(fn)
def voiced(c):
    min_s, max_s=25,146

    is_voiced = max(c[min_s:max_s]) > .2*c[0] # znelost se urcuje jako max(R)>a.R(0), a muze byt treba 0.1
    #is_voiced = max(c) > 0.2
    return is_voiced
def get_f0(c):
    #rozsah lagu ve vzorcich, pri 8000Hz odpovida
    min_s, max_s= 25,146
    #vyberu nejvyssi autokorelacni koeficient
    pitch = argmax(c[min_s:max_s])+min_s

    return pitch
def load_file(fn):
    out=[]
    with open(fn,'rb') as infile:
        data = numpy.fromfile(fn, dtype=numpy.uint8)
        #data=infile.read()#.split(" ")
        for idx in xrange(0,len(data)-3,3): #10*lpc+e+f0
            #out.append([float(d) for d in data[idx:idx+11]])
          #  print (data[idx])
            out.append(int(data[idx]))
            out.append(int(data[idx+1]))
            out.append(int(data[idx+2]))

    return (idx+3)/3,out # pocet vzorku a vystupni parametry

def decode(samples,enc,codebook_lpc,codebook_e):
    # axs.grid(True)
    #b, aa = butter(1, 200/ 8000., 'high')

    flt_f0=medfilt(enc[2::3],kernel_size=5)
    #print(enc[2::3])
    out=zeros(samples*(FRAME_SIZE-OVERLAP))
    for idx in xrange(0,len(enc),3):
        a_idx=enc[idx]
        e_idx=enc[idx+1]
        f0_sam=int(flt_f0[idx/3])#
        #f0_sam =enc[idx+2]
        print (f0_sam)
        if f0_sam !=0:
            vocoded = zeros(FRAME_SIZE)
            vocoded[f0_sam::f0_sam] = 1.0
            #vocoded =lfilter(b, aa,vocoded)
        else:
            vocoded = numpy.random.random(FRAME_SIZE)/4#???
        e=codebook_e[e_idx]
        a=codebook_lpc[a_idx]
        vocoded = lfilter((e,), a, vocoded)

        try:
            out[idx/3 * (FRAME_SIZE-OVERLAP):idx/3 * (FRAME_SIZE-OVERLAP) + FRAME_SIZE] += vocoded
        except Exception as e:
            print (e)

    return out

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
def encode(fn,codebook_lpc,codebook_e):
    out=zeros(len(data)*2)
    out=[]
    mean_sig=0
    pitches=[]
    #signal [0] je fs
    signal=scipy.io.wavfile.read(fn)[1]/2.0**15

    for idx, frame in enumerate(make_frames(signal)):#sf.frames(fn, framesize=FRAME_SIZE, overlap=OVERLAP)):
        #mean_sig=0.1*mean(frame)+0.9*mean_sig
        frame-=mean(signal)

        #autokorelacni koeficienty
        c=(correlate(frame, frame, mode='full'))
        c = c[len(c)/2:]
        f0_sam=get_f0(c)

        pitches.append(f0_sam)
        is_voiced=voiced(c)


        if not is_voiced:
            f0_sam=0

        #frame = preemphasis(frame)
        #rxx = correlate(frame,frame, mode='full')
        #rxx = rxx[len(rxx) / 2:]
        #a, k = levinson_algorithm(rxx[:LPC + 1])
        #e = sqrt(mean(lfilter(a, (1,), frame)**2))
        a,e,_=lpc(frame,LPC)
        e=sqrt(e)[0]



        # vezmeme z codebooku
        e_idx=vq(e,codebook_e)[0]
        a=a.reshape((1,LPC+1))
        a_idx=vq(a,codebook_lpc)[0]

       # e = rms(lfilter(a, (1,), frame))

        out.append(a_idx)
        out.append(e_idx)
        out.append(f0_sam)
        #fig, axs = plt.subplots(1, 1)
        #axs.plot(frame)
    return numpy.asarray(out)

codebook_lpc=load_codebook("codebook_lpc.npy")
codebook_e=load_codebook("codebook_gain.npy")

enc=encode("test.wav",codebook_lpc,codebook_e)
if True:
    with (open("out.cod",'wb')) as outfile:
        for idx in xrange(0, len(enc), 3):
            a_idx = enc[idx][0]
            e_idx = enc[idx + 1][0]
            f0_sam = enc[idx + 2]
            outfile.write(chr(int(a_idx)))
            outfile.write(chr(int(e_idx)))
            outfile.write(chr(int(f0_sam)))

    idx,data=load_file("out.cod")
    out=decode(idx,data,codebook_lpc,codebook_e)
    #print (load_file("out.cod"))
    wavwrite('vocoded.wav', 8000, array(out/amax(absolute(out)) * (2**15-1), dtype=int16))


