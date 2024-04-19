import parselmouth
import numpy as np
import soundfile as sf

class TimbreAnalyser:

  def __init__(self, audio_file):

    self.audio_array, self.samplerate = sf.read(audio_file)
    self.praat_soundobj = parselmouth.Sound(self.audio_array, self.samplerate)
    self.amps = None
    self.mfccs = None
    self.winstart = 0
    self.blocksize = 32 # vary this value? name is misleading - is really (self.blocksize*10) milliseconds  
    self.winend = None
    self.mfccs = None

  def get_amps(self):

    self.amps = self.praat_soundobj.to_intensity(time_step=0.01)
    self.amps = self.amps.values[0]

  def get_max_amp_window(self):

    start_point = 0
    windows = []

    for index, amplitude in enumerate(self.amps):
      if not index + (self.blocksize+1) > len(self.amps):
        windows.append(self.amps[start_point:start_point+self.blocksize])
        start_point += 1

    windows = np.array(windows)
    max_window = windows[0]

    for index, window in enumerate(windows[1:]):
      current_mean = float(np.mean(window))
      if current_mean > float(np.mean(max_window)):
        max_window = window
        self.winstart = index

    self.winend = self.winstart + self.blocksize

    print("mean of max amp window is:", float(np.mean(windows[self.winstart])))
    return self.winstart, self.winend

  def get_mean_amp_loudest_window(self):

    return np.mean(self.amps[self.winstart:self.winend])

  def get_mfccs_loudest_window(self):

    raw_startpos = int((self.winstart / len(self.amps)) * len(self.audio_array))
    raw_endpos = int((self.winend / len(self.amps)) * len(self.audio_array))

    mfcc_chunk = self.audio_array[raw_startpos:raw_endpos]

    smolchunk = parselmouth.Sound(mfcc_chunk, self.samplerate)
    self.mfccs = smolchunk.to_mfcc(number_of_coefficients=1)
    self.mfccs = self.mfccs.to_array()
    print("mfccs:", self.mfccs)
    return self.mfccs
