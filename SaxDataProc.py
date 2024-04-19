#!/usr/bin/env python

import os
import numpy as np
import librosa
import scipy.io.wavfile as wav
import soundfile as sf
import skimage.io as io
import sox
from shutil import rmtree, move
from pydub import AudioSegment, effects
from numpy import inf
import parselmouth
import torch
import torchaudio
from nnAudio import features
from rich import print
import statistics
import json

class SaxDataProcessing:

    def __init__(self, name: str, in_file: str, data_dir:str):

        self.classname = name
        self.in_file = in_file
        self.chunk_dir = './{}_chunks'.format(self.classname)
        self.aug_dir = './{}_aug'.format(self.classname)
        self.sax_train_dir = data_dir
        self.spec_dir = os.path.join(self.sax_train_dir, './{}'.format(self.classname))
        self.phrase_dir = './{}_phrases'.format(self.classname)
        self.norm_phrase_dir = './{}_norm_phrases'.format(self.classname)
        self.sr=22050
        self.pitches = None
        self.pitch_thresh = None
        self.amplitudes = None
        self.amp_thresh = None
        self.cqt = None

    def make_dirs(self):

        if not os.path.exists(self.sax_train_dir):
            os.mkdir(self.sax_train_dir)
        if not os.path.exists(self.chunk_dir):
            os.mkdir(self.chunk_dir)
        if not os.path.exists(self.aug_dir):
            os.mkdir(self.aug_dir)
        if not os.path.exists(self.spec_dir):
            os.mkdir(self.spec_dir)
        if not os.path.exists(self.phrase_dir):
            os.mkdir(self.phrase_dir)
        if not os.path.exists(self.norm_phrase_dir):
            os.mkdir(self.norm_phrase_dir)

        print('created directories')

    def resample_audio(self):
        data, sr = sf.read(self.in_file)
        # data = data.astype(np.float64)
        if sr != 22050:
            data = librosa.resample(data, sr, self.sr, 'polyphase')
            sf.write(self.in_file, samplerate=self.sr, data=data)
            print('resampled source audio')
        else:
            print('data is already at desired sr of 22050')
            pass

    def chunk_by_phrase(self):

        phrases_dict = {}

        audio_data = parselmouth.Sound(self.in_file)
        print('audio loaded')
        self.pitches = audio_data.to_pitch_ac(time_step=0.2, pitch_floor=50.0, pitch_ceiling=1400.0, very_accurate = True) # check this doesn't need a sr arg
        self.pitches = self.pitches.selected_array['frequency']
        self.pitches[self.pitches==0] = np.nan
        # self.pitches = list(self.pitches)
        self.pitches = np.nan_to_num(self.pitches)

        ### ADDED IN 02/06/22 - CONVERT TO MIDI FORMAT, NOT SURE IF IT'LL HELP ###
        self.pitches = 12*np.log2(self.pitches/440)+69
        self.pitches[self.pitches == -inf] = 0
        self.pitches = np.around(self.pitches, decimals=1)

        self.pitches=list(self.pitches)

        with open("phrase_test.txt", "w") as txtfile:
            txtfile.write(str(self.pitches))

        y, sr = sf.read(self.in_file)
        start = 0
        end = 0
        count = len(sorted(os.listdir(self.phrase_dir))) + 1

        for i in range(len(self.pitches)-1):
            if self.pitches[i] == 0 and self.pitches[i+1] >= 43.0:
                start = i
            if self.pitches[i] >= 43.0 and self.pitches[i+1] == 0:
                end = i+2
                # print(start, end)
                if int(int((end/5)*sr)-int((start/5)*sr)) >= 22050:
                    phrase = y[int((start/5)*sr)+int(0.1*sr):int((end/5)*sr)-int(0.1*sr)]
                    phrases_dict[str(count)] = self.pitches[start:end]
                    # phrase = phrase / np.max(np.abs(phrase))
                    sf.write(os.path.join(self.phrase_dir, '{}.wav'.format(count)), samplerate=sr, data=phrase)
                    count+=1
                # start = 0
                # end = 0
        print('chunked audio by phrase for playback later')

        with open("phrase_test.json", "w") as output_file:
            json.dump(phrases_dict, output_file)


    def normalize_phrases(self):

        for wavfile in os.listdir(self.phrase_dir):
            y, sr = sf.read(os.path.join(self.phrase_dir, wavfile))
            y = y / np.max(np.abs(y))
            sf.write(os.path.join(self.norm_phrase_dir, wavfile), samplerate=sr, data=y)


            # rawsound = AudioSegment.from_file(wavfile, "wav")
            # normalizedsound = effects.normalize(rawsound)
            # normalizedsound.export(os.path.join(self.phrase_dir, wavfile), format="wav")
        print('normalized phrases')

    def get_mean_pitch_thresh(self):

        self.pitches = np.array(self.pitches)
        self.pitches = 12*np.log2(self.pitches/440)+69
        self.pitches[self.pitches == -inf] = 0
        self.pitches = np.around(self.pitches, decimals=1)
        self.pitches = self.pitches[self.pitches!=0.0]
        self.pitch_thresh = self.pitches.mean()
        print('mean pitch is ', self.pitch_thresh)
        return self.pitch_thresh

    def group_phrases_by_mean_pitch(self, low_dir, high_dir):

        if not os.path.exists(low_dir):
            os.mkdir(low_dir)

        if not os.path.exists(high_dir):
            os.mkdir(high_dir)

        for wavfile in os.listdir(self.phrase_dir):

            audio_data = parselmouth.Sound(os.path.join(self.phrase_dir, wavfile))
            temp_pitches = audio_data.to_pitch_ac(time_step=0.1, pitch_floor=50.0, pitch_ceiling=1400.0) # check this doesn't need a sr arg
            temp_pitches = temp_pitches.selected_array['frequency']
            temp_pitches[temp_pitches==0] = np.nan
            temp_pitches = np.nan_to_num(temp_pitches)
            temp_pitches = 12*np.log2(temp_pitches/440)+69
            temp_pitches[temp_pitches == -inf] = 0
            temp_pitches = np.around(temp_pitches, decimals=1)
            temp_pitches = temp_pitches[temp_pitches!=0.0]
            print(temp_pitches)

            if temp_pitches.mean() <= self.pitch_thresh:
                move(os.path.join(self.phrase_dir, wavfile), os.path.join(low_dir, wavfile))
                print('moved file {} to low dir'.format(wavfile))

            if temp_pitches.mean() > self.pitch_thresh:
                move(os.path.join(self.phrase_dir, wavfile), os.path.join(high_dir, wavfile))
                print('moved file {} to high dir'.format(wavfile))

    def group_phrases_by_mean_amp(self, soft_dir, loud_dir, amp_thresh):

        self.amp_thresh = amp_thresh

        if not os.path.exists(soft_dir):
            os.mkdir(low_dir)

        if not os.path.exists(loud_dir):
            os.mkdir(high_dir)

        for wavfile in os.listdir(self.phrase_dir):

            audio_data = parselmouth.Sound(os.path.join(self.phrase_dir, wavfile))
            self.amplitudes = audio_data.to_intensity(time_step=0.01) # check this doesn't need a sr arg
            self.amplitudes = self.amplitudes.values
            self.amplitudes = np.ndarray.tolist(self.amplitudes)
            self.amplitudes = self.amplitudes[0]

            if statistics.mean(self.amplitudes) <= self.amp_thresh:
                move(os.path.join(self.phrase_dir, wavfile), os.path.join(soft_dir, wavfile))
                print('moved file {} to soft dir'.format(wavfile))

            if statistics.mean(self.amplitudes) > self.amp_thresh:
                move(os.path.join(self.phrase_dir, wavfile), os.path.join(loud_dir, wavfile))
                print('moved file {} to loud dir'.format(wavfile))

    def group_phrases_by_interval_size(self, in_dir, wide_dir, narrow_dir):

        if not os.path.exists(narrow_dir):
            os.mkdir(narrow_dir)

        if not os.path.exists(wide_dir):
            os.mkdir(wide_dir)

        for wavfile in os.listdir(in_dir):

            audio_data = parselmouth.Sound(os.path.join(in_dir, wavfile))
            temp_pitches = audio_data.to_pitch_ac(time_step=0.1, pitch_floor=50.0, pitch_ceiling=1400.0) # check this doesn't need a sr arg
            temp_pitches = temp_pitches.selected_array['frequency']
            temp_pitches[temp_pitches==0] = np.nan
            temp_pitches = np.nan_to_num(temp_pitches)
            temp_pitches = 12*np.log2(temp_pitches/440)+69
            temp_pitches[temp_pitches == -inf] = 0
            temp_pitches = np.around(temp_pitches, decimals=1)

            temp_onsets = np.ediff1d(temp_pitches) #or d = diff(midi)
            temp_onsets = (temp_onsets <= -0.8) & (temp_onsets >= -44) | (temp_onsets >= 0.8)
            temp_onsets = temp_onsets.astype(int)
            # replace consecutive onsets with 0:
            temp_onsets = list(temp_onsets)
            #print('temp onsets:', temp_onsets)
            final_onsets=[]
            for i in range(len(temp_onsets)-1):
                if temp_onsets[i] == 0:
                    final_onsets.append(temp_onsets[i])
                if temp_onsets[i] == 1 and temp_onsets[i+1] == 0:
                    final_onsets.append(temp_onsets[i])
                if temp_onsets[i] == 1 and temp_onsets[i+1] == 1:
                    final_onsets.append(0)

            final_onsets = np.insert(final_onsets, 0, 0)
            #print('onsets', self.onsets)
            temp_pitches = final_onsets * temp_pitches[:-1]

            nz = np.flatnonzero(temp_pitches)
            if max(temp_pitches) > 44:
                temp_pitches= temp_pitches[nz[0]:] # this threw error
                print('extracted onsets')
            else:
                pass

            temp_pitches = temp_pitches[temp_pitches!=0]
            intervals = np.ediff1d(temp_pitches)
            intervals = np.abs(intervals)
            print(intervals)
            mean_interval = (intervals.sum()/len(temp_pitches))
            print('mean interval is ', mean_interval)

            if intervals.mean() >= 4:
                move(os.path.join(in_dir, wavfile), os.path.join(wide_dir, wavfile))
                print('moved file {} to wide dir'.format(wavfile))

            if intervals.mean() < 4:
                move(os.path.join(in_dir, wavfile), os.path.join(narrow_dir, wavfile))
                print('moved file {} to narrow dir'.format(wavfile))

    def add_effects_to_phrases(self, in_dir, out_dir, effect='distortion'):

        # for wavfile in os.listdir(self.phrase_dir):
        #     tfm = sox.Transformer()
        #     tfm.downsample(2)
        #     tfm.build_file(os.path.join(self.phrase_dir, wavfile), os.path.join(self.phrase_dir, wavfile))
        if effect=='bandpass':

            for wavfile in os.listdir(self.phrase_dir):
                snd = AudioSegment.from_wav(os.path.join(self.phrase_dir, wavfile))
                snd = effect.band_pass_filter(snd, 2500, 10000, order=5)
                snd.export(os.path.join(self.phrase_dir, wavfile), format='wav')

        if effect=='distortion':

            for wavfile in os.listdir(in_dir):
                tfm = sox.Transformer()
                tfm.overdrive(colour=60)
                tfm.build_file(os.path.join(in_dir, wavfile), os.path.join(out_dir, wavfile))

        print('applied {} effect to {} and saved to {}'.format(effect, in_dir, out_dir))

    def truncate_silence(self):
        startPos = 0
        thresh = 15 # needs to be high bc audio has been normalized
        snd_array = []
        slnt_array = []
        array, sr = sf.read(self.in_file)
        chunk_len = int(sr/5)
        endPos=int(sr/5)

        try:

            for i in range(0, (len(array)-chunk_len), chunk_len): # I know this is horrible but the func still runs fast
                if np.mean(np.abs(array[startPos:endPos])) > thresh:
                    snd_array.append(array[startPos:endPos])
                if np.mean(np.abs(array[startPos:endPos])) < thresh:
                    slnt_array.append(array[startPos:endPos])
                startPos+=chunk_len
                endPos+=chunk_len

            snd_array = np.concatenate(snd_array).ravel()
            sf.write(self.in_file, samplerate=sr, data=snd_array)
            print('removed silences')

            slnt_array = np.concatenate(slnt_array).ravel()
            sf.write(self.silent_file, samplerate=sr, data=slnt_array)
            print('wrote silence file')

        except ValueError as e:
            pass

    def data_augmentation_pitchshift(self): # data augmentation through pitch shift
        tfm1 = sox.Transformer()
        tfm2 = sox.Transformer()
        tfm1.pitch(1.0)
        tfm1.build_file(self.in_file, os.path.join(self.aug_dir, 'aug1.wav'))
        tfm2.pitch(-1.0)
        tfm2.build_file(self.in_file, os.path.join(self.aug_dir, 'aug2.wav'))

        combined = []

        for file in os.listdir(self.aug_dir):
            sr, sound = wav.read(os.path.join(self.aug_dir, file))
            combined.append(sound)

        sr, sound2 = wav.read(self.in_file)
        combined.append(sound2)

        combined=np.hstack(combined)

        wav.write(self.in_file, rate=sr, data=combined.astype(np.int16)) # not sure why i converted to int16 here
        print('augmented data')

    def data_augmentation_timestretch(self):

        data, sr = sf.read(self.in_file)

        fast_data = librosa.effects.time_stretch(data, 1.1)
        slow_data = librosa.effects.time_stretch(data, 0.95)

        sf.write(os.path.join(self.aug_dir, "faster.wav"), samplerate = sr, data = fast_data)
        sf.write(os.path.join(self.aug_dir, "slower.wav"), samplerate = sr, data = slow_data)

        print("saved augmented data")

        data, sr = sf.read(self.in_file)
        fast_data, sr = sf.read(os.path.join(self.aug_dir, "faster.wav"))
        slow_data, sr = sf.read(os.path.join(self.aug_dir, "slower.wav"))

        concatenated = np.hstack((data, fast_data))
        concatenated = np.hstack((concatenated, slow_data))

        sf.write(self.in_file, samplerate = sr, data = concatenated)

        print("concatenated and saved augmented data")

    def chunk_train_audio(self):
        chunk_len = 32768
        startpos = 0
        endpos = 32768
        count=0
        sr, data = wav.read(self.in_file)
        for i, n in enumerate(data):
            if i % chunk_len == 0:
                if len(data) - startpos >= 32768:
                    count+=1
                    chunk = data[startpos:endpos]
                    chunk = chunk/np.max(np.abs(chunk))
                    wav.write(os.path.join(self.chunk_dir,'{}.wav'.format(str(count).zfill(6))), sr, chunk)
                    startpos = (startpos+chunk_len)
                    endpos = (endpos+chunk_len)
                else:
                    break

        print('chunked audio')

    def compute_CQTs(self):
        spec_count = 0
        for wavfile in sorted(os.listdir(self.chunk_dir)):
            sr, y = wav.read(os.path.join(self.chunk_dir, wavfile))
            y = y.astype(np.float64)
            # res_type='polyphase' should speed this up
            cqt = np.abs(librosa.cqt(y,  sr=sr, hop_length=512, fmin=64, n_bins=64, bins_per_octave=12, sparsity=0.01, res_type='polyphase'))
            cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)
            spec_count+=1
            io.imsave(os.path.join(self.spec_dir, '{}.jpg'.format(str(spec_count).zfill(6))), cqt_db)
        print('computed spectros')

    def compute_salient_CQTs(self):
        spec_count = 0
        for wavfile in sorted(os.listdir(self.chunk_dir)):
            y, sr = sf.read(os.path.join(self.chunk_dir, wavfile))
            cqt = np.abs(librosa.cqt(y,  sr=sr, hop_length=512, fmin=64, n_bins=64, bins_per_octave=12, sparsity=0.01, res_type='polyphase'))
            # cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)
            freqs = librosa.cqt_frequencies(n_bins = 64, fmin = 64)
            harms = [1, 2, 3, 4]
            weights = [1.0, 0.5, 0.33, 0.25]
            salient = librosa.salience(cqt, h_range = harms, freqs = freqs, weights = weights, fill_value = 0)
            spec_count+=1
            io.imsave(os.path.join(self.spec_dir, '{}.jpg'.format(str(spec_count).zfill(6))), salient)
        print('computed spectros')

    def compute_MFCCs(self):
        spec_count = 0
        for wavfile in sorted(os.listdir(self.chunk_dir)):
            sr, y = wav.read(os.path.join(self.chunk_dir, wavfile))
            y = y.astype(np.float64)
            # res_type='polyphase' should speed this up
            mfcc = librosa.feature.mfcc(y,  sr=sr, hop_length=512, n_mfcc=64)
            spec_count+=1
            io.imsave(os.path.join(self.spec_dir, '{}.jpg'.format(str(spec_count).zfill(6))), mfcc)
        print('computed spectros')

    def get_CQT_layer(self):

        sr, y = wav.read(self.in_file)
        y = torch.FloatTensor(y)
        self.cqt = features.CQT(hop_length=512, fmin=64, n_bins=64, bins_per_octave=12)
        return self.cqt

    def compute_CQTs_GPU(self):

        for chunk in os.listdir(self.chunk_dir):
            sr, y = wav.read(os.path.join(self.chunk_dir, chunk))
            y = torch.FloatTensor(y)
            cqt_spec = self.cqt(y)
            cqt_spec = torch.abs(cqt_spec)
            cqt_spec = torchaudio.transforms.AmplitudeToDB(top_db=80)(cqt_spec)
            cqt_spec = cqt_spec.cpu().detach().numpy()
            # cqt_spec = cqt_spec.reshape((64, 64))
            cqt_spec = cqt_spec.reshape((cqt_spec.shape[1], cqt_spec.shape[2]))
            io.imsave(os.path.join(self.spec_dir, chunk[:-4]+'.jpg'), cqt_spec)

    def cleanup(self):

        rmtree(self.aug_dir)
        rmtree(self.chunk_dir)
        #os.remove(self.in_file)
