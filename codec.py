import scipy,numpy,math
from scipy.io.wavfile import read
from scipy.cluster.vq import kmeans2
import soundfile as sf
from scipy.signal import hann, lfilter, freqz, decimate, butter
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite
from scipy.signal import hann, lfilter, freqz, decimate, butter
from numpy import array, double, amax, absolute, zeros, floor, arange, mean
from numpy import correlate, dot, append, divide, argmax, int16, sqrt, power
from numpy.random import randn
from scikits.talkbox.linpred import lpc
import matplotlib
import matplotlib.pyplot as plt


# tuhle fci potom muzeme pouzit na trenovani codebooku pro LPC gain
sample_rate,data=read("test.wav")
print (sample_rate)
data=data.astype('float32')
print (data)
clusters=kmeans2(data, 10)
#print (clusters)
print (data.shape)



for idx, block in enumerate(sf.blocks('test.wav', blocksize=256, overlap=128)):
    #odstraneni SS
    block-=mean(block)

    fig, axs = plt.subplots(1, 1)
    axs.plot(block)

    axs.grid(True)

    fig.tight_layout()
    plt.show()
    print (lpc(block,10))






