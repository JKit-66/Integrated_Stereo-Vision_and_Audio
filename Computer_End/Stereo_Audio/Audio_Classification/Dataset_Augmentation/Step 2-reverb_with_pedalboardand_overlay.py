from pydub import AudioSegment
#from torchaudio.utils import download_asset
from pedalboard import *
#from pedalboard import Pedalboard, Chorus, Reverb, Compressor, Gain, Phaser
from pedalboard.io import AudioFile
import os
import random
import numpy as np
import scipy.signal
import soundfile as sf
import librosa

class AudioEditor():
    def __init__(self):
        self.noise_direct = 'C:/Users/J Kit/Desktop/Y3 Intern/Week4-DistanceModelPhysical/monday/OverlayNoises'
        self.main_file_parent_dir = 'C:/Users/J Kit/Desktop/Y3 Intern/Week6-AudioAugmentation-一条龙/2-trimmedaudio'
        self.main_audio_fileN = ''
        self.sr = 44100
        self.duration = 1.0
        self.list_of_bgnoise = self.give_noise_file_direct(self.noise_direct)
        self.list_of_dataset = self.give_file_direct(self.main_file_parent_dir)
        
    def give_noise_file_direct(self, direct):
        editable_file = []
        main_files = os.listdir(direct)

        for main_file in main_files:
            main_file_dir = os.path.join(direct, main_file)
            editable_file.append(main_file_dir)
        
        return editable_file
    
    def give_file_direct(self, direct):
        editable_file = []
        main_files = os.listdir(direct)
        for main_file in main_files:
            main_file_dir = os.path.join(direct, main_file)
            sub_files = os.listdir(main_file_dir)

            for sub_file in sub_files:
                editable_file.append(os.path.join(main_file_dir, sub_file))
        return editable_file 

    def reverB(self, main_file, fileN):
        szize = [0.1, 0.4, 0.6, 0.7, 0.05, 0.8]
        for size in szize:
            board = Pedalboard([Chorus(), Reverb(room_size=float(size))])
            # Open an audio file for reading, just like a regular file:
            with AudioFile(main_file) as f:
                # Open an audio file to write to:
                real_dir_fileN = os.path.dirname(fileN)
                real_fileN = f'{size}_{os.path.basename(fileN)}'
                with AudioFile(os.path.join(real_dir_fileN, real_fileN), 'w', f.samplerate, f.num_channels) as o:
                    # Read one second of audio at a time, until the file is empty:
                    while f.tell() < f.frames:
                        chunk = f.read(f.samplerate)
                        # Run the audio through our pedalboard:
                        effected = board(chunk, f.samplerate, reset=False)
                        # Write the output to our output file:
                        o.write(effected)

    def chain_ReverB(self, main_file, fileN):
        y, sr = librosa.load(main_file, sr=None)
        #print(sr)
        board = Pedalboard([Compressor(ratio=10, threshold_db=-20),
                            Gain(gain_db=20),
                            Phaser(),
                            Reverb()])
        effected = board.process(y, sr)
        samplerate = 44100.0

        with AudioFile(main_file).resampled_to(samplerate) as f:
            audio = f.read(f.frames)

            # Make a pretty interesting sounding guitar pedalboard:
            board = Pedalboard([
                Compressor(ratio=10, threshold_db=-20),
                Gain(gain_db=20),
                Phaser(),
                Reverb()
            ])

            # ... or change parameters easily:
            board[0].threshold_db = -40

            # Run the audio through this pedalboard!
            effected = board(audio, samplerate)

            # Write the audio back as a wav file:
            real_dir_fileN = os.path.dirname(fileN)
            real_fileN = f'Chain_{os.path.basename(fileN)}'
            with AudioFile(os.path.join(real_dir_fileN, real_fileN), 'w', samplerate, effected.shape[0]) as f:
                f.write(effected)

    def overLAY(self, main_file, bckgndnoise_audio_fileN, fileN):
        background_noise = AudioSegment.from_file(bckgndnoise_audio_fileN)
        main_audio = AudioSegment.from_file(main_file)
        # If the background noise is shorter than the main audio, loop it
        if 'BckgNoise_apply' in bckgndnoise_audio_fileN:
            background_noise = background_noise - 35
        elif '_rain' in bckgndnoise_audio_fileN:
            background_noise = background_noise - 45
        else:
            background_noise = background_noise - 5
        
        while len(background_noise) < len(main_audio):
            background_noise += background_noise

        # Overlay the background noise onto the main audio
        combined_audio = main_audio.overlay(background_noise)

        # Export the result to a new file
        real_dir_fileN = os.path.dirname(fileN)
        real_fileN = f'ovly_{os.path.basename(fileN)}'
        fileN_fin = os.path.join(real_dir_fileN, real_fileN)
        combined_audio.export(fileN_fin , format="wav")
    

    def randomly_delete_files(self, directory, target_count=2500):
        # Get a list of all files in the directory
        files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        
        while len(files) > 2500:
            # Calculate the number of files to delete
            num_files_to_delete = len(files) - target_count
            
            if num_files_to_delete <= 0:
                print(f"The directory already has {len(files)} files, which is less than or equal to the target count of {target_count}.")
                return

            # Randomly select files to delete
            files_to_delete = random.sample(files, num_files_to_delete)

            # Delete the selected files
            for file in files_to_delete:
                filebase =  os.path.basename(file)
                if len(os.path.splitext(filebase)[0].split('_')) > 12:
                    prob = [0,1,2]
                    pickle = random.choice(prob)
                    if pickle == 1:
                        os.remove(file)
                        #print(f"Deleted: {file}")
                else:
                    prob = [0,1,2,3,4,5,7]
                    pickle = random.choice(prob)
                    if pickle == 1:
                        os.remove(file)
                        #print(f"Deleted: {file}")
                
            files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

        print(f"Deleted {num_files_to_delete} files. {target_count} files remain.")

if __name__ == '__main__':
    editor = AudioEditor()
    all_of_files = editor.list_of_dataset
    num =0 
    all_of_classes = []
    
    phone_clss = ['bello-fone'] #['iphone', 'telefon'] #['bellfone', 'iphone', 'telefon']
    clss = ['drone', 'keyboard']
    dic_clss = {'drone': 0, 'keyboard': 0, 'bellfone': 0, 'iphone': 0, 'telefon': 5, 'jogging': 5}
    lis = [0,1,2]
    noisesfileN = ['C:/Users/J Kit/Desktop/Y3 Intern/Week4-DistanceModelPhysical/monday/OverlayNoises/0_0.wav', 'C:/Users/J Kit/Desktop/Y3 Intern/Week4-DistanceModelPhysical/monday/OverlayNoises/1_1.wav', 'C:/Users/J Kit/Desktop/Y3 Intern/Week4-DistanceModelPhysical/monday/OverlayNoises/BckgNoise_rain.wav']
    for vid in all_of_files:
        curr_class = os.path.basename(os.path.dirname(vid))
        

        if curr_class not in all_of_classes:
            all_of_classes.append(curr_class)
        
        print(all_of_classes)

        if curr_class == 'jogging':
            for i in range(dic_clss[curr_class]):
                print(i)
                bckgndnoise_audio_fileN = random.choice(editor.list_of_bgnoise)
                editor.main_audio_fileN = vid
                
                #main_audio = AudioSegment.from_file(vid)
                fileN = os.path.join(os.path.dirname(vid), os.path.splitext(os.path.basename(bckgndnoise_audio_fileN))[0] + '_-_' + os.path.splitext(os.path.basename(vid))[0]+'.wav')

                fileN_permn = os.path.basename(vid)
                ok = os.path.splitext(fileN_permn)[0].split('_')[0]
            
                try:
                    if ('BckgNoise' in bckgndnoise_audio_fileN) or ('0_' in bckgndnoise_audio_fileN) or ('1_' in bckgndnoise_audio_fileN) :
                        editor.overLAY(vid, bckgndnoise_audio_fileN, fileN)
                        for filesNoise in noisesfileN:
                            editor.overLAY(vid, filesNoise, fileN)
                            print('OVERLAY--')
                        #print(fileN)
                    elif 'RIR_' in bckgndnoise_audio_fileN:
                        ok_choice = random.choice(lis)
                        print('RIR--')
                        if ok_choice == 3:
                            editor.chain_ReverB(vid, fileN)
                        elif (ok_choice == 1) or (ok_choice == 2):
                            editor.chain_ReverB(vid, fileN)
                            editor.reverB(vid, fileN)
                        else:
                            editor.reverB(vid, fileN)
                except Exception as e:
                    print(e)


        #if curr_class not in phone_clss:
            
                # bckgndnoise_audio_fileN = random.choice(editor.list_of_bgnoise)
                # editor.main_audio_fileN = vid
                # curr_class = os.path.basename(os.path.dirname(vid))


    for classs in all_of_classes:
        parent_dir = 'C:/Users/J Kit/Desktop/Y3 Intern/Week6-AudioAugmentation-一条龙/2-trimmedaudio'
        directoriez = os.path.join(parent_dir, classs)
        editor.randomly_delete_files(directoriez)


    print('@@@## Step 6 Done !')