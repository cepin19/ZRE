import scipy,numpy,math
from scipy.io.wavfile import read
from scipy.cluster.vq import kmeans2
import soundfile as sf
from scipy.signal import hann, lfilter, freqz, decimate, butter
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite
from scipy.signal import hann, lfilter, freqz, decimate, butter
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

    is_voiced = max(c[min_s:max_s]) > 0.25*c[0] # znelost se urcuje jako max(R)>a.R(0), a muze byt treba 0.1
    return is_voiced
def get_f0(c):

   # print (c)
    #rozsah lagu ve vzorcich, pri 8000Hz odpovida f 50-400Hz
    min_s, max_s= 20,160
    #vyberu nejvyssi autokorelacni koeficient
    pitch = argmax(c[min_s:max_s])
    #print(c)

    return pitch

b, aa = butter(1, 200/8000., 'high')

out=zeros(len(data)*2)
for idx, block in enumerate(sf.blocks('test.wav', blocksize=256, overlap=128)):
    glottal_lowpass = lambda signal: lfilter(b, aa, block)

    #odstraneni SS

    block-=mean(block)
    #spocitani autokorelacnich koeficientu
    block = (block - mean(block)) / (std(block) * len(block))
    #autokorelacni koeficienty
    c=(correlate(block, block, mode='full'))
    c = c[len(c)/2:]
    f0_sam=get_f0(block)
    is_voiced=voiced(c)
    f0_freq=1/(f0_sam/8000.)
    print (idx)
    print (f0_sam)
    print (f0_freq)
    print (is_voiced)
    a,e,_=lpc(block,10)
    e=e**2
    print (a)
    print ("______________________")

    #fig, axs = plt.subplots(1, 1)
    #axs.plot(block)

    #axs.grid(True)
    if is_voiced and f0_sam!=0:
        vocoded = zeros(len(block))
        vocoded[f0_sam::f0_sam] = 1.0
        vocoded = glottal_lowpass(vocoded)
    else:
        vocoded = randn(len(block))/2
        vocoded = lfilter(e, a, vocoded)
        vocoded *= hann(len(block))
    try:
        out[idx*256:idx*256+256] += vocoded
    except Exception as e:
        print (e)

    #return out
    #fig.tight_layout()
    #plt.show()
    #print (lpc(block,10))

print(array(out) * (2**15-1))

wavwrite('vocoded.wav', 8000, array(out/amax(absolute(out)) * (2**15-1), dtype=int16))


