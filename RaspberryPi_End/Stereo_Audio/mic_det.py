import numpy as np

class Ears():
    def __init__(self):
        self.channel = 1
        self.sr = 44100
        self.duration = 1
    
    def calculate_spl(self, y):
        
        reference_pressure = 20e-6  #In air, the common reference is 20 μPa, ISBN-10: 0-309-05025-1
        rms = np.sqrt(np.mean(np.square(y)))
        max_ = np.max(y)

        # Calculate SPL in decibels
        spl = 20 * np.log10(rms / reference_pressure)
        return spl, max_
    
    def get_max_amplitude(self, amplitude):
        #return np.mean(np.abs(amplitude))
        return np.max(np.abs(amplitude))

    def compare_amplitudes(self, amplitude1, amplitude2):
        # Read the audio files
        #amplitude1 = self.read_wave_file(file1)
        #amplitude2 = self.read_wave_file(file2)
        
        # Calculate the average amplitude of each file
        max_amplitude1 = amplitude1
        max_amplitude2 = amplitude2
        
        # Compare the average amplitudes
        if max_amplitude1 > max_amplitude2:
            return 'L', max_amplitude1/max_amplitude2
        else:
            return 'R', max_amplitude1/max_amplitude2

    def distance_estimation(self, average_SPL):
        return round((average_SPL/128.81) ** (-1/0.01), 3)
      
    def direction_estimation(self, time_delay):
        #print(time_delay)
        return round(((time_delay*10000)/0.0448) *0.98, 3)
