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
FRAME_SIZE=256
OVERLAP=128
LPC=16
numpy.set_printoptions(threshold=numpy.nan)

# tuhle fci potom muzeme pouzit na trenovani codebooku pro LPC gain
sample_rate,data=read("test.wav")
print (sample_rate)
data=data.astype('float32')
print (data)
clusters=kmeans2(data, 10)
#print (clusters)
print (data.shape)

#TODO neuklada 0. lpc koef



fs=8000



def levinson_algorithm(r):
    """
    Calculates A and K coefficients of an auto correlation array using
    the levinson-durben method.
    """
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
def fundamental_period_estimate(rxx,fs):
    """
    Calculates the fundamental frequency of an auto correlation array.
    rxx   the auto correlation array.
    fs    the sample rate in hertz.
    """
    f_low, f_high = 50 , 250
    f_low_idx = int(round(fs / f_low))
    f_high_idx = int(round(fs / f_high))
    period_idx = argmax(rxx[f_high_idx:f_low_idx ]) + f_high_idx
    is_voiced = max(rxx) > 0.20
    return(period_idx, is_voiced)

def train_codebook(train_files):
    lpcs=[]
    es=[]
    pitches=[]
    for fn in train_files:
        for idx, block in enumerate(sf.blocks(fn, blocksize=FRAME_SIZE, overlap=OVERLAP)):
            gain_correction = (1 - OVERLAP) * 2  # *2 due to hann window
            block *= hann(len(block)) * gain_correction
            block -= mean(block)
            # spocitani autokorelacnich koeficientu
            # autokorelacni koeficienty
            c = (correlate(block, block, mode='full'))
            c = c[len(c) / 2:]
            f0_sam = get_f0(c)
           # print ("old")
            #print f0_sam

            #f0_sam = medfilt(pitches, kernel_size=3)[-1]
            #print ("new")
            #print f0_sam
            is_voiced = voiced(c)
            f0_freq = 1 / (f0_sam / 8000.)
            #rxx = correlate(block, block, mode='full')
            #rxx = rxx[len(rxx) / 2:]
            #f0_sam, is_voiced = fundamental_period_estimate(rxx, fs)

            if not is_voiced:
                f0_sam = 0

            #block = preemphasis(block)
            #rxx = correlate(block, block, mode='full')
            #rxx = rxx[len(rxx)/2:]
            #a, k = levinson_algorithm(rxx[:LPC+1])
            #e = rms(lfilter(a, (1,), block))

            a, e, _ = lpc(block, LPC)
            e = math.sqrt(e)
            lpcs.append(a)
            es.append(e)
            pitches.append(float(f0_sam))

    # v test.wav je 265 vzorku, takze by mely centroidy odpovidat jednotlivym hodnotam presne
    clusters_lpc = kmeans(lpcs, 256)[0]
    clusters_e = kmeans(es, 256)[0]
    clusters_f0 = kmeans(pitches, 256)[0]

    print (clusters_lpc)

    return clusters_lpc,clusters_e,clusters_f0

def voiced(c):
    min_s, max_s=50,250

    is_voiced = max(c[min_s:max_s]) > 0.5*c[min_s] # znelost se urcuje jako max(R)>a.R(0), a muze byt treba 0.1
    #is_voiced = max(c) > 0.0001
    return is_voiced
def get_f0(c):

   # print (c)
    #rozsah lagu ve vzorcich, pri 8000Hz odpovida f 50-400Hz
    min_s, max_s= 50,250
    #vyberu nejvyssi autokorelacni koeficient
    pitch = argmax(c[min_s:max_s])+min_s
    #print(c)

    return pitch
def load_file(fn):
    out=[]
    with open(fn,'rb') as infile:
        data = numpy.fromfile(fn, dtype=numpy.uint8)
        #data=infile.read()#.split(" ")
        for idx in xrange(0,len(data)-3,3): #10*lpc+e+f0
            #out.append([float(d) for d in data[idx:idx+11]])
            print (data[idx])
            out.append(int(data[idx]))
            out.append(int(data[idx+1]))
            out.append(int(data[idx+2]))
            print (out)
            #exit()
           # print ("aa")
            #print (numpy.asarray(data[idx:idx+10],dtype='float32'))
            #print ("e")
            #print ([float(data[idx+11])])
            #print ("f0")
            #print (int(float(data[idx+12])))
    return (idx+3)/3,out # pocet vzorku a vystupni parametry

def decode(samples,enc,codebook_lpc,codebook_e,codebook_f0):
    # axs.grid(True)
    #print(len(enc))
    b, aa = butter(1, 100/ 8000., 'high')

    out=zeros(samples*OVERLAP)
    for idx in xrange(0,len(enc),3):
        #print(idx)
        a_idx=enc[idx]
        e_idx=enc[idx+1]
        #numpy.ndarray((1,),buffer=enc[idx+1],dtype='float32')
        f0_sam_idx=enc[idx+2]

        #print (a)
        #print (e)
        #print (f0_sam)
        f0_sam=int(codebook_f0[f0_sam_idx])
        print (f0_sam)
        if f0_sam !=0:
            print ("voiced")
            vocoded = zeros(FRAME_SIZE)
            #f0_sam=
            vocoded[f0_sam::f0_sam] = 1.0
            #???
            vocoded =lfilter(b, aa,vocoded)
        else:
            print ("unvoiced")
            vocoded = randn(FRAME_SIZE)
        e=codebook_e[e_idx]
        a=codebook_lpc[a_idx]
        print (e)
        print (a)
        vocoded = lfilter((e,), a, vocoded)
        vocoded *= hann(FRAME_SIZE)
        #print (vocoded.shape)
        #print (out.shape)

        try:
          #  print ("starti %s"%(idx/3 * OVERLAP))
         #   print ("endi %s"%((idx/3 * OVERLAP)+FRAME_SIZE))

            out[idx/3 * OVERLAP:idx/3 * OVERLAP + FRAME_SIZE] += vocoded
        except Exception as e:
            print (e)

    return out
            # fig.tight_layout()
            # plt.show()
            # print (lpc(block,10))


def encode(fn,codebook_lpc,codebook_e,codebook_f0):
    out=zeros(len(data)*2)
    out=[]
    mean_sig=0
    pitches=[]
    for idx, block in enumerate(sf.blocks(fn, blocksize=FRAME_SIZE, overlap=OVERLAP)):
        gain_correction = (1 - OVERLAP) * 2  # *2 due to hann window
        block *= hann(len(block)) * gain_correction
        #odstraneni SS
        mean_sig=0.5*mean(block)+0.5*mean_sig
        block-=mean(block)
        #spocitani autokorelacnich koeficientu
        #autokorelacni koeficienty
        c=(correlate(block, block, mode='full'))
        c = c[len(c)/2:]
        f0_sam=get_f0(c)

        pitches.append(f0_sam)
        #print ("old")
        #print f0_sam
        f0_sam=medfilt(pitches,kernel_size=3)[-1]
        #print ("new")
        #print f0_sam

        is_voiced=voiced(c)
       # rxx = correlate(block, block, mode='full')
        #rxx = rxx[len(rxx) / 2:]
        #f0_sam, is_voiced = fundamental_period_estimate(rxx, fs)

        f0_freq=1/(f0_sam/8000.)




        if not is_voiced:
            f0_sam=0
            f0_freq=0

       # print (idx)
       # print (f0_sam)
       # print (f0_freq)
       # print (is_voiced)
       # block = preemphasis(block)


        a,e,_=lpc(block,LPC)
        e=math.sqrt(e)

        #block = preemphasis(block)
        #rxx = correlate(block, block, mode='full')
        #rxx = rxx[len(rxx) / 2:]
        #a, k = levinson_algorithm(rxx[:LPC + 1])
        #e = rms(lfilter(a, (1,), block))

        # vezmeme z codebooku
        e_idx=vq(e,codebook_e)[0]
        print (a.shape)
        print (codebook_lpc.shape)
        a=a.reshape((1,LPC+1))
        a_idx=vq(a,codebook_lpc)[0]
        f0_idx=vq(f0_sam,codebook_f0)[0]

        #print (a)
        #print (e)

       # e = rms(lfilter(a, (1,), block))



        #print ("______________________")
        out.append(a_idx)
        out.append(e_idx)
        out.append(f0_idx)
        #fig, axs = plt.subplots(1, 1)
        #axs.plot(block)
    return numpy.asarray(out)
(codebook_lpc,codebook_e,codebook_f0)=train_codebook(["test.wav"])
print (codebook_lpc)

print (codebook_lpc.shape)
print (codebook_e.shape)
#print(array(out) * (2**15-1))
enc=encode("test.wav",codebook_lpc,codebook_e,codebook_f0)
if True:
    with (open("out.cod",'wb')) as outfile:
        for idx in xrange(0, len(enc), 3):
            print(idx)
            a_idx = enc[idx][0]
            #outfile.write(str(a_idx)+ " ")
            #for aa in a:
            #    outfile.write( "{:.9f} ".format(float(aa)))

            e_idx = enc[idx + 1][0]
            print (e_idx)
            #outfile.write(str(e_idx)+' ')

            f0_idx = enc[idx + 2][0]
            #outfile.write(str(f0_sam) +" ")
            outfile.write(chr(int(a_idx)))
            outfile.write(chr(int(e_idx)))
            outfile.write(chr(int(f0_idx)))
           # outfile.write("\n")

           # print ("a")
           # print (a)
           # print ("e")
           # print (e)
           # print ("f0")
           # print (f0_sam)
    #print (load_file("out.cod"))
    idx,data=load_file("out.cod")
    out=decode(idx,data,codebook_lpc,codebook_e,codebook_f0)
    #print (load_file("out.cod"))
    wavwrite('vocoded.wav', 8000, array(out/amax(absolute(out)) * (2**15-1), dtype=int16))


