import parselmouth
import numpy as np
from numpy import inf
import os

class PitchAnalyser:

    def __init__(self, audio_file):

        self.audio_file = audio_file
        self.audio_data = None
        self.pitches = None
        self.onsets = None
        self.first_pitch = None
        self.num_pitches = None
        self.last_pitch = None

    def load_audio(self):

        self.audio_data = parselmouth.Sound(self.audio_file)
        print('audio loaded')

    def get_freqs(self):

        self.pitches = self.audio_data.to_pitch_ac(time_step=0.01, pitch_floor=50.0, pitch_ceiling=1400.0, very_accurate = True) # check this doesn't need a sr arg
        self.pitches = self.pitches.selected_array['frequency']
        self.pitches[self.pitches==0] = np.nan
        self.pitches = list(self.pitches)
        self.pitches = np.nan_to_num(self.pitches)
        print('extracted freqs')

    def freqs_to_MIDI(self):

        self.pitches = 12*np.log2(self.pitches/440)+69
        self.pitches[self.pitches == -inf] = 0
        self.pitches = np.around(self.pitches, decimals=1)
        print('converted freqs to MIDI')
        print(self.pitches)

    def get_onsets(self):

        # work out which note values represent an onset and multiply the two vectors
        temp_onsets = np.ediff1d(self.pitches) #or d = diff(midi)

        temp_onsets = (temp_onsets <= -0.8) & (temp_onsets >= -44) | (temp_onsets >= 0.8)
        temp_onsets = temp_onsets.astype(int)

        # replace consecutive onsets with 0:
        temp_onsets = list(temp_onsets)
        #print('temp onsets:', temp_onsets)
        self.onsets=[]

        # for i, n in enumerate(temp_onsets):
        #     if n == 0:
        #         self.onsets.append(n)
        #     if n == 1 and temp_onsets[i+1] == 0:
        #         self.onsets.append(temp_onsets[i])
        #     if n == 1 and temp_onsets[i+1] == 1 and i != len(temp_onsets) -1):
        #         self.onsets.append(0)

        for i in range(len(temp_onsets)-1):

            if temp_onsets[i] == 0:
                self.onsets.append(temp_onsets[i])
            if temp_onsets[i] == 1 and temp_onsets[i+1] == 0:
                self.onsets.append(temp_onsets[i])
            if temp_onsets[i] == 1 and temp_onsets[i+1] == 1:
                self.onsets.append(0)

        self.onsets = np.insert(self.onsets, 0, 0)
        self.onsets = np.insert(self.onsets, len(self.onsets-1), 0)
        #print('onsets', self.onsets)
        self.pitches = self.onsets * self.pitches
        print(self.pitches)
        print(max(self.pitches))
        nz = np.flatnonzero(self.pitches)
        if max(self.pitches) > 44:
            self.pitches= self.pitches[nz[0]:] # this threw error
            print('extracted onsets')
        else:
            pass
        return self.pitches
        print(self.pitches)

    def remove_zeros_for_pitches_only(self):

        self.pitches = self.pitches[self.pitches!=0]
        print("after zeros removed:", self.pitches)


    def float2int(self):

        print(type(self.pitches))

        for index, num in enumerate(self.pitches):
            self.pitches[index] = int(num)

        print("converted floats to integers")

    def return_outputs(self):

        self.first_pitch = self.pitches[0]
        self.last_pitch = self.pitches[-1]
        self.num_pitches = self.pitches.size

        return self.first_pitch, self.last_pitch, self.num_pitches
