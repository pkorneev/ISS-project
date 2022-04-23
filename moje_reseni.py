import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.io import wavfile
from scipy.signal import spectrogram, freqz, tf2zpk
from scipy.signal import find_peaks
from scipy import signal

#1 Zaklady

fs, data = wavfile.read('xkorni03.wav')
z = np.arange(data.size) / fs
samples = str(len(data))
freqency = str(fs)
length_in_sec = str(z.max())
print("Freqency: " + freqency)
print("Length in samples: " + samples)
print("Min & max: " + str(data.min()) + ", " + str(data.max()))
print("Length in seconds: " + length_in_sec)

#Freqency: 16000
#Length in samples: 47616
#Min & max: -6207, 10939
#Length in seconds: 2.9759375

plt.figure(figsize=(15,8))
plt.plot(z,data)
plt.ylabel("Amplitude")
plt.xlabel("Time (s) ")
plt.title("Zvukový signál")
plt.show()
plt.savefig('signal.pdf')

#2 Predzpracovanı a ramce
mean = np.mean(data)  # stredni hodnota
data = data - mean       
normalize = np.abs(data) # absolutni hodnota
data = data / (normalize.max())

frame = []
for i in range(0, data.size, 512):
    frame.append(data[i: i + 1024])

n = 14
data_value = frame[n]
z = np.arange(n*1024,data_value.size+n*1024)/fs

plt.figure(figsize=(15,8))
plt.plot(z,data_value)
plt.ylabel("Amplitude")
plt.xlabel("Time (s)")
plt.show()
plt.savefig("norm_frame.pdf".format(x=n))

#3 DFT  
#we need 1024 samples

#odkud = 2
#kolik = 0.064 #1024 samples

#odkud_vzorky = int(odkud * fs)         # start of segment in samples
#pokud_vzorky = int((odkud+kolik) * fs) # end of segment in samples
#first one fft from lib
rangeee = np.arange(0,fs/2,fs/2/(data_value.size/2))
fft = np.fft.fft(data_value)
plt.figure(figsize=(15,8))
plt.plot(rangeee,np.abs(fft[:int(fft.size/2)]))
plt.ylabel("Amplitude")
plt.xlabel("Frequency (Hz)")
plt.title("FFT from np.lib")
plt.show()
plt.savefig("fft.pdf")
# dft now
dft = np.zeros(data_value.size, complex)

for k in range(0, data_value.size):
    for n in range(0, data_value.size):
        dft[k] += data_value[n] * np.exp(-1j * 2*np.pi*k*n/data_value.size)

rangeee = np.arange(0,fs/2,fs/2/(data_value.size/2))
plt.figure(figsize=(15,8))
plt.plot(rangeee,np.abs(dft[:int(dft.size/2)]))
plt.ylabel("Amplitude")
plt.xlabel("Frequency (Hz)")
plt.title("DFT") 
plt.show()
plt.savefig("dft.pdf")

#4 Spektrogram

f, t, sgr = spectrogram(data, fs ,nperseg=1024,noverlap=512)
sgr_log = 10 * np.log10(sgr+1e-20) 

plt.figure(figsize=(10,6))
plt.pcolormesh(t,f,sgr_log)
plt.gca().set_xlabel('Time (s)')
plt.gca().set_ylabel('Frequency (Hz)')
cbar = plt.colorbar()
cbar.set_label('Spectral power density (dB)', rotation=270, labelpad=15)
plt.tight_layout()
plt.show()
plt.savefig("spectrogram.pdf")


#5 Urcenı rusivych frekvencı

frame = []
for i in range(f.size ):
    frame.append(sgr_log[i][0])

peaks, _ = scipy.signal.find_peaks(frame,height=-65)
print(peaks) # [ 50 100 150 200]
# we have 4 peaks (4 max indexes) , now we need to konvert them into a frequency 
freqencies = np.asarray(f[peaks])
print(freqencies) #[ 781.25 1562.5  2343.75 3125.  ]

#6 Generovanı signalu

def create_signal(freq, sampling_rate): 
    time = np.arange(float(length_in_sec) * sampling_rate) / sampling_rate
    return np.cos(2 * np.pi * freq * time)

signal_1 = create_signal(freqencies[0], fs)
signal_2 = create_signal(freqencies[1], fs)
signal_3 = create_signal(freqencies[2], fs)
signal_4 = create_signal(freqencies[3], fs)
signals = signal_1 + signal_2 + signal_3 + signal_4

f, t, sgr = spectrogram(signals, fs, nperseg=1024,noverlap=512)
sgr_log = 10 * np.log10(sgr+1e-20) 

plt.figure(figsize=(10,6))
plt.pcolormesh(t,f,sgr_log)
plt.gca().set_xlabel('Time (s)')
plt.gca().set_ylabel('Frequency (Hz)')
cbar = plt.colorbar()
cbar.set_label('Spectral power density (dB)', rotation=270, labelpad=15)

plt.tight_layout()
plt.show()
plt.savefig("signal_spectrogram.pdf")

wavfile.write("4cos.wav", fs, signals) # be careful in headphones with thisone XD