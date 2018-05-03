import scipy,numpy,math
from scipy.io.wavfile import read
from scipy.cluster.vq import kmeans2
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

numpy.set_printoptions(threshold=numpy.nan)

# tuhle fci potom muzeme pouzit na trenovani codebooku pro LPC gain
sample_rate,data=read("test.wav")
print (sample_rate)
data=data.astype('float32')
print (data)
clusters=kmeans2(data, 10)
#print (clusters)
print (data.shape)



def voiced(c):
    min_s, max_s= 20,160

    is_voiced = max(c[min_s:max_s]) > 0.15*c[0] # znelost se urcuje jako max(R)>a.R(0), a muze byt treba 0.1
    return is_voiced
def get_f0(c):

   # print (c)
    #rozsah lagu ve vzorcich, pri 8000Hz odpovida f 50-400Hz
    min_s, max_s= 20,160
    #vyberu nejvyssi autokorelacni koeficient
    pitch = argmax(c[min_s:max_s])+min_s
    #print(c)

    return pitch
def load_file(fn):
    out=[]
    with open(fn) as infile:
        data=infile.read().split(" ")
        for idx in xrange(0,len(data)-14,13): #10*lpc+e+f0
            out.append([float(d) for d in data[idx:idx+11]])
            out.append([float(data[idx+11])])
            out.append(int(float(data[idx+12])))
           # print ("aa")
            #print (numpy.asarray(data[idx:idx+10],dtype='float32'))
            #print ("e")
            #print ([float(data[idx+11])])
            #print ("f0")
            #print (int(float(data[idx+12])))
    return out#numpy.asarray(out,dtype='float32')

def decode(enc):
    # axs.grid(True)
    #print(len(enc))
    b, aa = butter(1, 800 / 8000., 'high')

    out=zeros(len(data) * 2)
    for idx in xrange(0,len(enc),3):
        #print(idx)
        a=numpy.asarray(enc[idx])
        e=numpy.asarray(enc[idx+1])
        #numpy.ndarray((1,),buffer=enc[idx+1],dtype='float32')
        f0_sam=enc[idx+2]

        #print (a)
        #print (e)
        #print (f0_sam)

        if f0_sam != 0:
            print ("voiced")
            vocoded = zeros(256)
            #f0_sam=
            vocoded[f0_sam::f0_sam] = 1.0
            #???
#            vocoded =lfilter(b, aa,vocoded)
        else:
            print ("unvoiced")
            vocoded = randn(256)
        vocoded = lfilter(e, a, vocoded)
        vocoded *= hann(256)
        #print (vocoded.shape)
        #print (out.shape)

        try:
          #  print ("starti %s"%(idx/3 * 128))
         #   print ("endi %s"%((idx/3 * 128)+256))

            out[idx/3 * 128:idx/3 * 128 + 256] += vocoded
        except Exception as e:
            print (e)

    return out
            # fig.tight_layout()
            # plt.show()
            # print (lpc(block,10))


def encode(fn):
    out=zeros(len(data)*2)
    out=[]
    pitches=[]
    for idx, block in enumerate(sf.blocks(fn, blocksize=256, overlap=128)):
        gain_correction = (1 - 128) * 2  # *2 due to hann window
        block *= hann(len(block)) * gain_correction
        #odstraneni SS

        block-=mean(block)
        #spocitani autokorelacnich koeficientu
        #autokorelacni koeficienty
        c=(correlate(block, block, mode='full'))
        c = c[len(c)/2:]
        f0_sam=get_f0(c)
        pitches.append(f0_sam)
        print ("old")
        print f0_sam
        f0_sam=medfilt(pitches,kernel_size=3)[-1]
        print ("new")
        print f0_sam

        is_voiced=voiced(c)

        f0_freq=1/(f0_sam/8000.)




        if not is_voiced:
            f0_sam=0
            f0_freq=0

       # print (idx)
       # print (f0_sam)
       # print (f0_freq)
       # print (is_voiced)
       # block = preemphasis(block)
   

        a,e,_=lpc(block,10)
       # e=math.sqrt(e**2)
        #print (a)
        #print (e)

       # e = rms(lfilter(a, (1,), block))



        #print ("______________________")
        out.append(a)
        out.append(e)
        out.append(f0_sam)
        #fig, axs = plt.subplots(1, 1)
        #axs.plot(block)
    return numpy.asarray(out)

#print(array(out) * (2**15-1))
enc=encode("test.wav")
with (open("out.cod",'w')) as outfile:
    for idx in xrange(0, len(enc), 3):
        print(idx)
        a = enc[idx]
        for aa in a:
            outfile.write( "{:.9f} ".format(float(aa)))

        e = enc[idx + 1]
        outfile.write("{:.9f} ".format(float(e)))

        f0_sam = enc[idx + 2]
        outfile.write(str(f0_sam) +" ")
        outfile.write("\n")

       # print ("a")
       # print (a)
       # print ("e")
       # print (e)
       # print ("f0")
       # print (f0_sam)
#print (load_file("out.cod"))
out=decode(load_file("out.cod"))
#print (load_file("out.cod"))
wavwrite('vocoded.wav', 8000, array(out*2/amax(absolute(out)) * (2**15-1), dtype=int16))


