from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import os
import soundfile as sf
import librosa
from peakdetect import peakdetect

def get_y_value_at_x(data, x_value):
    # Get the y value at the specified x value
    y_value = data[x_value]

    return y_value

def get_rms(values):
    values = np.array(values)
    squared_values = values ** 2
    mean_squared_values = np.mean(squared_values)
    rms_value = np.sqrt(mean_squared_values)
    return rms_value


def time_shift_editor(file_path, shift_max, shift_direction):
        #data, sample_rate = sf.read(file_path)
        data, _ = librosa.load(file_path,sr=None)
        sample_rate = 44100
        shift = np.random.randint(sample_rate * shift_max)
        if shift_direction == 'right':
            shift = -shift
        elif shift_direction == 'both':
            direction = np.random.randint(0, 2)
            if direction == 1:
                shift = -shift
        augmented_data = np.roll(data, shift)
        if shift > 0:
            augmented_data[:shift] = 0
        else:
            augmented_data[shift:] = 0
        
        #save_dir = C:/Users/J Kit/Desktop/Y3 Intern/Week3-AudioDataAugmentation/lews\1\guitar_21k.wav
        sf.write('C:/Users/J Kit/Desktop/Y3 Intern/Week7-Audio_Direction_callibrate/csv_result/new_8_0.wav', augmented_data, sample_rate)

validity = 'P' #'T2'
R = 'C:/Users/J Kit/Desktop/Y3 Intern/Week7-TimeShift_AUDIO/17_0.wav'
L = 'C:/Users/J Kit/Desktop/Y3 Intern/Week7-TimeShift_AUDIO/new_17_1.wav'
L_O = 'C:/Users/J Kit/Desktop/Y3 Intern/Week7-Audio_Direction_callibrate/L_R_020824_test/17.449_DEG_440.411_L/0_1.wav'


#L2 = time_shift_editor(R, 0.003696, 'right')  #0.003696

rateL, dataL = wavfile.read(L)
rateR, dataR = wavfile.read(R)
rateL0, dataL0 = wavfile.read(L_O)

peaksL = peakdetect(dataL, lookahead=180) 
higherPeaksL = np.array(peaksL[0])
lowerPeaksL = np.array(peaksL[1])

peaksR = peakdetect(dataR, lookahead=180) 
higherPeaksR = np.array(peaksR[0])
lowerPeaksR = np.array(peaksR[1])

lengthL = dataL.shape[0] / rateL
timeL = np.linspace(0., lengthL, dataL.shape[0])

lengthL0 = dataL0.shape[0] / rateL0
timeL0 = np.linspace(0., lengthL0, dataL0.shape[0])

lengthR = dataR.shape[0] / rateR
timeR = np.linspace(0., lengthR, dataR.shape[0])

#start, end = array_split([0.375, 0.382], rate)
#plt.plot(dataL, label="Audio Left")
#plt.plot(timeL0, dataL0, label=f"Audio {os.path.basename(L_O)}_augmented")
#plt.plot(dataR, label="Audio Right")

#plt.plot(higherPeaksR[:,0], higherPeaksR[:,1], 'rx')
#plt.plot(lowerPeaksR[:,0], lowerPeaksR[:,1], 'kx')
#plt.plot(higherPeaksL[:,0], higherPeaksL[:,1], 'rx')
#plt.plot(lowerPeaksL[:,0], lowerPeaksL[:,1], 'kx')

plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
#plt.show()

if validity == 'P':
    #cleaned_clip_1 = 'C:/Users/J Kit/Desktop/Y3 Intern/Week7-TimeShift_AUDIO/05082024/4sec/after/clean_1.wav'
    #cleaned_clip_0 = 'C:/Users/J Kit/Desktop/Y3 Intern/Week7-TimeShift_AUDIO/05082024/4sec/after/clean_0.wav'

    #cleaned_clip_0 = '@@0808_C/1sec/2/1sec/1_0.wav'
    #cleaned_clip_1 = '@@0808_C/1sec/2/1sec/1_1.wav'

    cleaned_clip_0 = 'report_writing/18_0.wav'
    cleaned_clip_1 = 'report_writing/18_1.wav'
    
    #L_b4 = "C:/Users/J Kit/Desktop/Y3 Intern/Week7-Audio_Direction_callibrate/05082024_F/sound/41.72_1360.425_R_DEG_10/0_1.wav"
    #R_b4 = "C:/Users/J Kit/Desktop/Y3 Intern/Week7-Audio_Direction_callibrate/05082024_F/sound/41.72_1360.425_R_DEG_10/0_0.wav"

    L_b4 = "C:/Users/J Kit/Desktop/Y3 Intern/Week7-Audio_Direction_callibrate/05082024_F/sound/R_#_R_#_R_#_R/1_1.wav"
    R_b4 = "C:/Users/J Kit/Desktop/Y3 Intern/Week7-Audio_Direction_callibrate/05082024_F/sound/R_#_R_#_R_#_R/1_0.wav"

    # L is 1, R is 0 #####
    rateL, dataL = wavfile.read(cleaned_clip_1)
    rateR, dataR = wavfile.read(cleaned_clip_0)
    
    lengthL = dataL.shape[0] / rateL
    timeL = np.linspace(0., lengthL, dataL.shape[0])
    lengthR = dataR.shape[0] / rateR
    timeR = np.linspace(0., lengthR, dataR.shape[0])

    rateL_b4, dataL_b4 = wavfile.read(L_b4)
    rateR_b4, dataR_b4 = wavfile.read(R_b4)

    #plt.plot(dataL_b4, label="Audio Left_B4")
    #plt.plot(dataR_b4, label="Audio Right_B4")

    plt.plot(dataR, label="Audio from Right Microphone", color='blue')
    plt.plot(dataL, label="Audio from Left Microphone", color='red')
    

    plt.legend()
    plt.title('Waveform of Signals vs Time, with Sound Source from Right Side') 
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()


if validity == 'T':
    nancy = 0
    x_pt = []
    y_pt = []
    y_pt_PP = []
    #x : higherPeaksR[:,0], y : higherPeaksR[:,1]
    for x_loc in higherPeaksR[:,0]:
        L_Loud = get_y_value_at_x(dataL, x_loc)
        R_Loud = get_y_value_at_x(dataR, x_loc)
        print(f'For L Audio, at {x_loc}, the y is {L_Loud}')
        print(f'For R Audio, at {x_loc}, the y is {R_Loud}')

        if abs(L_Loud) > abs(R_Loud):
            nancy += 1
            x_pt.append(x_loc)
            y_pt.append(L_Loud)
            y_pt_PP.append(R_Loud)

    if nancy >= int(len(higherPeaksR[:,0]) * 0.5):
        print('Left Louder')
    else:
        print('Right Louder')

    print(nancy)

    plt.plot(dataR, label="Audio Right")
    plt.plot(dataL, label="Audio Left")
    plt.plot(x_pt, y_pt, label="Marker_L")
    plt.plot(x_pt, y_pt_PP, label="Marker_R")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()


if validity == 'T2':
    nancy = 0
    x_pt = []
    y_pt_PP = []

    combinedPeaksR = np.concatenate((lowerPeaksR[:, 0], higherPeaksR[:, 0]))
    combinedPeaksL = np.concatenate((lowerPeaksL[:, 0], higherPeaksL[:, 0]))
    combinedPeaksR = combinedPeaksR[combinedPeaksR.argsort()]
    combinedPeaksL = combinedPeaksL[combinedPeaksL.argsort()]


    for x_loc in combinedPeaksR:
        R_Loud = get_y_value_at_x(dataR, x_loc)
        x_pt.append(x_loc)
        y_pt_PP.append(np.abs(R_Loud))

    x_pt2 = []
    y_pt2 = []
    for x_loc in combinedPeaksL:
        L_Loud = get_y_value_at_x(dataL, x_loc)
        x_pt2.append(x_loc)
        y_pt2.append(np.abs(L_Loud))
    
    plt.plot(dataR, label="Audio Right")
    plt.plot(dataL, label="Audio Left")

    #plt.plot(x_pt, y_pt, label="Marker_L, RHP")
    
    plt.plot(x_pt, y_pt_PP, '--', label="Marker_R, RP")
    plt.plot(x_pt2, y_pt2, '--', label="Marker_L, LP")

    #plt.plot(x_pt_L, y_pt_PP_L, label="Marker_R, RLP")
    #plt.plot(x_pt2_L, y_pt2_L, label="Marker_L, LLP")

    #plt.plot(x_pt2, y_pt_PP2, label="Marker_R, LHP")

    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    #plt.show()

    print(get_rms(y_pt_PP), get_rms(y_pt2))
