import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.io import wavfile
#Amplitude is inversely proportional to distance
#init_amp in dB

def calculate_spl(y):
    # Reference sound pressure (typically 20 µPa in air)
    reference_pressure = 20e-6  #In air, the common reference is 20 μPa, ISBN-10: 0-309-05025-1
    
    # Calculate RMS (Root Mean Square) of the audio signal
    rms = np.sqrt(np.mean(np.square(y)))

    print(rms)
    # Calculate SPL in decibels
    spl = 20 * np.log10(rms / reference_pressure)
    return spl

def audio_amplitude_estimate(init_dist, init_amp, final_dist):
    # use 10log10() refers to a measure for acoustic energy
    # use 20log10() for amplitude
    p2_p1 = 10**(init_amp/10)
    gap = final_dist/init_dist
    final_p2p1 = (p2_p1/gap**2)
    log_final_p2p1 = 10*np.log10(final_p2p1)
    
    intensity = 1e-12 * 10**(log_final_p2p1/10)
    return log_final_p2p1,intensity

# Load an audio file
file_path = 'C:/Users/J Kit/Desktop/Y3 Intern/Week3-MethodStatement&RiskAssessment/guitar_16k.wav'
y, sr = librosa.load(file_path)

# Calculate and print SPL
spl = calculate_spl(y)
print(f'Sound Pressure Level: {spl} dB')


ampli_list =[]
dist_list = []
intensity = []
#distance = np.arange(1, 5, 0.25)
distance = 1
amplitude = 100
while distance <= 6:
    distance += 0.25
    indB, _ = audio_amplitude_estimate(1,spl,distance)
    amplitude = indB
    ampli_list.append(indB)
    dist_list.append(distance)
    #for j in distance:
    #intensity.append(intens)

plt.plot(dist_list, ampli_list, '.-')
plt.xlabel('Distance from Sound Source(m    )')
plt.ylabel('Amplitude of Sound Source (dB)')
plt.grid()
plt.show()

plt.plot(distance, intensity, '.-')
plt.xlabel('Distance from Sound Source')
plt.ylabel('Intensity of Sound Source')
plt.grid()
#plt.show()