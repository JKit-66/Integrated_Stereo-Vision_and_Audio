from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

L_ = '0_1.wav'
R_ = '0_0.wav'

rateL, dataL = wavfile.read(L_)
rateR, dataR = wavfile.read(R_)

L_l = 'L1_0_1.wav'
R_l = 'L1_0_0.wav'

rateL, dataL_l = wavfile.read(L_l)
rateR, dataR_l = wavfile.read(R_l)


#plt.plot(dataR_l, 'c--', label='Right_L')
#plt.plot(dataL_l, 'b--', label='Left_L')


plt.plot(dataR, 'r--', label='Right')
plt.plot(dataL, 'k--', label='Left')

plt.legend()
plt.show()
