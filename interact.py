#PVT_Phrases_inference.py
import Recorder
from Recorder import Recorder
from SaxDataProc import SaxDataProcessing as SDP
import CNNModel_Exps as CNN
from PitchExtraction import PitchAnalyser
from TimbralExtraction import TimbreAnalyser
from Filter import Filter

from pydub import AudioSegment
from pydub.playback import _play_with_simpleaudio

import numpy as np

import time
import random
import os
import json

####################### LOAD CLASSIFIER ###############################

infer_sax = CNN.Inference(model_path = './sax_model_exp_17_02_2023_40eps_largestTestSet.pth', rec_path = './infer.wav', spec_path = './infer_sax.jpg', spec_type='cqt')
infer_sax.load_model()

###################### READ IN JSON FILES FOR PHRASE LOOKUPS #################

with open("v2_p.json") as lookup_file:
    pitch_phrase_lookup = json.load(lookup_file)
#
# with open("timbre_phrase_lookup.json") as lookup_file:
#     timbre_phrase_lookup = json.load(lookup_file)

############################ INTERACTIVE LOOP ###########################

t_count=0
p_count=0
amp_thresh=35

while True:

    rec = Recorder(channels=1)
    with rec.open('./infer.wav', 'wb') as recfile:
        recfile.record(duration=1.5)
    time.sleep(0.05)
    rec.resample_audio()
    time.sleep(0.05)
    rec.truncate_pad()
    time.sleep(0.05)

    filter = Filter(in_file='./infer.wav')
    loudness = filter.get_mean_amp()
    if loudness < amp_thresh:
        print('no saxophone input detected')
        pass

    elif loudness >= amp_thresh:
        infer_sax.compute_salient_CQT()
        sax_predict, sax_prob = infer_sax.infer_class()

        if sax_predict == 'pitch_data_salient':

            print("pitch")
            p_count += 1
            t_count=0

            print('p:', p_count, 't:', t_count)

            if p_count >= 2:

                matches = []

                pa = PitchAnalyser('infer.wav')

                pa.load_audio()
                pa.get_freqs()
                pa.freqs_to_MIDI()
                pa.get_onsets()
                pa.remove_zeros_for_pitches_only()
                pa.float2int()
                first, last, num_pitches = pa.return_outputs()

        # match last note i just played to first of playback

                for key, values in pitch_phrase_lookup.items():
                    if values[0] == last:
                        matches.append(key)

                if len(matches) == 0:
                    for i in range(-3, 3, 1):
                        for key, values in pitch_phrase_lookup.items():
                            if (values[0] + i) == last:
                                matches.append(key)

                if len(matches) == 0:
                    print("failed to find a match")
                    pass

                if len(matches) == 1:

                    snd = AudioSegment.from_file(os.path.join("v2_p_phrases", matches[0]))
                    snd = snd.fade_in(50).fade_out(50)
                    _play_with_simpleaudio(snd)
                    time.sleep(0.5)

                if len(matches) > 1:

                    snd = AudioSegment.from_file(os.path.join("v2_p_phrases", random.choice(matches)))
                    snd = snd.fade_in(50).fade_out(50)

                    # snd1 = AudioSegment.from_file(os.path.join("pitch_phrases", matches[0]))
                    # snd2 = AudioSegment.from_file(os.path.join("pitch_phrases", matches[1]))
                    # combined = snd1.append(snd2, crossfade=50)
                    # combined = combined.fade_in(50).fade_out(50)
                    # _play_with_simpleaudio(combined)
                    _play_with_simpleaudio(snd)

                    time.sleep(0.5)


                # p_count=0
                # t_count=0

            else:
                pass

        if sax_predict == 'timbre_data_salient':

            print("timbre")

            # if sax_prob > 0.85:

            t_count += 1
            p_count=0
            print('p:', p_count,'t:', t_count)

            if t_count >= 2:
            # if sax_prob >= 0.7:
                # t_count = 0
                #
                snd = AudioSegment.from_file(os.path.join("v2_t_phrases", random.choice(os.listdir("v2_t_phrases"))))
                snd = snd.fade_in(50).fade_out(50)
                _play_with_simpleaudio(snd)
                time.sleep(0.75)

                # matches = []
                #
                # rec.normalize()
                # time.sleep(0.05)
                #
                # ta = TimbreAnalyser('infer.wav')
                # ta.get_amps()
                # ta.get_max_amp_window()
                # mean_amp = ta.get_mean_amp_loudest_window()
                # # mfccs = ta.get_mfccs_loudest_window()
                # # print(mfccs)
                #
                # # for key, value in timbre_phrase_lookup.items():
                # #     if value == np.mean(mfccs).astype(np.int16):
                # #         matches.append(key)
                #
                # for key, value in timbre_phrase_lookup.items():
                #     if int(value) == int(mean_amp):
                #         matches.append(key)
                #
                # # if len(matches) == 0:
                # #
                # #     for key, value in timbre_phrase_lookup.items():
                # #         for i in range(-150, 150, 1):
                # #             if value == np.mean(mfccs).astype(np.int16):
                # #                 matches.append(key)
                # #
                # # for key, value in timbre_phrase_lookup.items():
                # #     for i in range(-1, 1, 1):
                # #         if value == np.mean(mfccs).astype(np.int16):
                # #             matches.append(key)
                #
                # if len(matches) == 0:
                #     print("failed to find a match")
                #     pass
                #
                # if len(matches) == 1:
                #
                #     snd = AudioSegment.from_file(os.path.join("timbre_norm_phrases", matches[0]))
                #     snd = snd.fade_in(50).fade_out(50)
                #     _play_with_simpleaudio(snd)
                #     time.sleep(0.75)
                #
                # if len(matches) > 1:
                #
                #     snd1 = AudioSegment.from_file(os.path.join("timbre_norm_phrases", random.choice(matches)))
                #     # snd2 = AudioSegment.from_file(os.path.join("timbre_norm_phrases", matches[1]))
                #     # combined = snd1.append(snd2, crossfade=50)
                #     snd1 = snd1.fade_in(50).fade_out(50)
                #     _play_with_simpleaudio(snd1)
                #     time.sleep(0.75)

        else:
            pass
    else:
        pass
