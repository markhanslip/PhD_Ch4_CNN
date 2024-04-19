import Recorder
from Recorder import Recorder
from SaxDataProc import SaxDataProcessing as SDP
from CNN import CNN 
from PitchExtraction import PitchAnalyser
from TimbralExtraction import TimbreAnalyser

import numpy as np

import json
import os
import time

# ########## IF RECORDING PHRASES FROM SCRATCH #####################
#
# rec1 = Recorder(channels=1)
# with rec1.open('./pitch_phrases.wav', 'wb') as recfile:
#     recfile.start_recording()
#     print('recording')
#     time.sleep(180*2)
#     recfile.stop_recording()
#     print('stopped recording')
#
# time.sleep(3)

# rec2 = Recorder(channels=1)
# with rec2.open('./timbre_phrases.wav', 'wb') as recfile:
#     recfile.start_recording()
#     print('recording')
#     time.sleep(180.0*2)
#     recfile.stop_recording()
#     print('stopped recording')
#
# time.sleep(2)

# ################# TRAIN DATA PREPROCESSING ####################

 train_proc1 = SDP(name='pitch_data_salient', in_file='./melodic_improv.wav', data_dir='./sax_train_data')
 train_proc1.make_dirs()
 train_proc1.resample_audio()
 train_proc1.truncate_silence()
 train_proc1.data_augmentation_pitchshift()
 # train_proc1.data_augmentation_timestretch()
 train_proc1.chunk_train_audio()
 train_proc1.compute_salient_CQTs()
 train_proc1.cleanup()

 train_proc2 = SDP(name='timbre_data_salient', in_file='./timbral_improv.wav', data_dir='./sax_train_data')
 train_proc2.make_dirs()
 train_proc2.resample_audio()
 train_proc2.truncate_silence()
 # train_proc2.data_augmentation_pitchshift()
 train_proc2.data_augmentation_timestretch()
 train_proc2.chunk_train_audio()
 train_proc2.compute_salient_CQTs()
 train_proc2.cleanup()

################# PHRASE CHUNKING #################

 phrase_proc1 = SDP(name='v2_p', in_file = './v2_p_mix.wav', data_dir = './sax_train_data')
 phrase_proc1.make_dirs()
 phrase_proc1.resample_audio()
 phrase_proc1.chunk_by_phrase()
 phrase_proc1.normalize_phrases()

 phrase_proc1 = SDP(name='v2_t', in_file = './v2_t_mix.wav', data_dir = './sax_train_data')
 phrase_proc1.make_dirs()
 phrase_proc1.resample_audio()
 phrase_proc1.chunk_by_phrase()
 phrase_proc1.normalize_phrases()

###################### MODEL TRAINING ###################

 saxtrain = CNN.Trainer(data_path='./sax_train_data/')
 saxtrain.calculate_mean_std()
 saxtrain.load_data()
 saxtrain.build_model()
 saxtrain.train(epochs=40)
 saxtrain.save_model(model_path='./sax_model_exp_17_02_2023_40eps_largestTestSet.pth')

# ############### BUILD HASHTABLE OF PHRASE INFO + SAVE TO JSON ###############

pitch_phrase_lookup = {}

for wavfile in os.listdir("v2_p_phrases"):

    try:

        pa = PitchAnalyser(os.path.join("v2_p_phrases", wavfile))

        pa.load_audio()
        pa.get_freqs()
        pa.freqs_to_MIDI()
        pa.get_onsets()
        pa.remove_zeros_for_pitches_only()
        pa.float2int()
        first, last, num_pitches = pa.return_outputs()

        pitch_phrase_lookup[wavfile] = (first, last, num_pitches)

    except IndexError as e:
        print("skipped one")
        pass

with open("v2_p.json", "w") as output_file:
    json.dump(pitch_phrase_lookup, output_file)

timbre_amp_lookup = {}

# ALTERNATIVE USING AMPLITUDE VALUES INSTEAD OF MFCC:

for wavfile in os.listdir("v2_t_phrases"):

    ta = TimbreAnalyser(os.path.join("v2_t_phrases", wavfile))

    ta.get_amps()
    ta.get_max_amp_window()
    mean_amp = ta.get_mean_amp_loudest_window()

    timbre_amp_lookup[wavfile] = mean_amp

with open("v2_t.json", "w") as output_file:
    json.dump(timbre_amp_lookup, output_file)
