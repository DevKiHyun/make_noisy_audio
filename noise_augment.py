import argparse
import numpy
import random
import os
import threading
import time
import math
import glob
import soundfile
from scipy import signal
from scipy.io import wavfile

def loadWAV(filepath, max_audio, evalmode=False, num_eval=10):
    """
    Ignore 'evalmode' and 'num_eval' argument. We not use in this code.
    """

    # Read wav file and convert to torch tensor
    audio, sample_rate = soundfile.read(filepath)
    audiosize = audio.shape[0]

    if audiosize <= max_audio:
        shortage    = max_audio - audiosize + 1 
        audio       = numpy.pad(audio, (0, shortage), 'wrap') # repeat wav to extend the wav length to max length
        audiosize   = audio.shape[0]

    if evalmode:
        startframe = numpy.linspace(0,audiosize-max_audio,num=num_eval)
    else:
        startframe = numpy.array([numpy.int64(random.random()*(audiosize-max_audio))])
    
    feats = []
    if evalmode and max_frames == 0:
        feats.append(audio)
    else:
        for asf in startframe:
            feats.append(audio[int(asf):int(asf)+max_audio])

    feat = numpy.stack(feats,axis=0).astype(numpy.float)

    return feat;


class AugmentWAV(object):
    def __init__(self, data_list, dest_dir, musan_path, rir_path, log_interval=100, **kwargs):
        self.data_list = data_list
        self.dest_dir = dest_dir
        self.log_interval = log_interval

        self.noisetypes = ['noise','speech','music']

        self.noisesnr   = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
        self.numnoise   = {'noise':[1,1], 'speech':[3,7],  'music':[1,1] }
        self.noiselist  = {}

        augment_files   = glob.glob(os.path.join(musan_path,'*/*/*/*.wav'));

        for file in augment_files:
            if not file.split('/')[-4] in self.noiselist:
                self.noiselist[file.split('/')[-4]] = []
            self.noiselist[file.split('/')[-4]].append(file)

        self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'));

    def run(self):
        count = 0 
        for i, data_path in enumerate(self.data_list):
            count += 1

            filename = os.path.basename(data_path) # get string 'filename.ext' from 'src_dir/filename.ext'
            dest_path = os.path.join(self.dest_dir, filename) # make 'dest_dir/filename.ext'

            audio, sample_rate = soundfile.read(data_path)

            aug_audio = self.augment_wav(audio)

            if len(aug_audio.shape) >= 2:
                aug_audio = aug_audio.squeeze(0)

            soundfile.write(dest_path, aug_audio, sample_rate)

            if i % self.log_interval == 0:
                print(f'{count}/{len(self.data_list)}...')


    def augment_wav(self, audio):
        augtype = random.randint(0,4) #  augment audio with 4 noise type randomly.
        if augtype == 1:
            audio   = self.reverberate(audio)
        elif augtype == 2:
            audio   = self.additive_noise('music',audio)
        elif augtype == 3:
            audio   = self.additive_noise('speech',audio)
        elif augtype == 4:
            audio   = self.additive_noise('noise',audio)

        return audio


    def additive_noise(self, noisecat, audio):
        max_audio = audio.shape[0]
        clean_db = 10 * numpy.log10(numpy.mean(audio ** 2)+1e-4) 

        numnoise    = self.numnoise[noisecat]
        noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))

        noises = []

        for noise in noiselist:
            noiseaudio  = loadWAV(noise, max_audio, evalmode=False)
            noise_snr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio[0] ** 2)+1e-4) 
            noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noiseaudio)

        return numpy.sum(numpy.concatenate(noises,axis=0),axis=0,keepdims=True) + audio


    def reverberate(self, audio):
        max_audio = audio.shape[0]

        rir_file    = random.choice(self.rir_files)
        
        rir, fs     = soundfile.read(rir_file)
        rir         = numpy.expand_dims(rir.astype(numpy.float),0)
        rir         = rir / numpy.sqrt(numpy.sum(rir**2))
        
        rirsize = rir.shape[1]

        if rirsize >= max_audio: # If rir size is longer than target audio length
            rir = rir[:,:max_audio]

        return signal.convolve(numpy.expand_dims(audio,0), rir, mode='full')[:,:max_audio]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filestxt',       default='filelist.txt', type=str, help='The txt with absolute path of files')
    parser.add_argument('--dest-dir',       default=None, type=str, help='The destination save path of augmented audio file')
    parser.add_argument('--musan-path',     default='musan_split/', type=str, help='musan file directory')
    parser.add_argument('--rir-path',       default='simulated_rirs/', type=str, help='rir file directory')
    parser.add_argument('--log-interval',   default=50,   type=int)
    args = parser.parse_args()

    with open(args.filestxt, "r") as f:
            args.data_list = f.read().splitlines()

    os.makedirs(args.dest_dir, exist_ok=True)

    augmentation = AugmentWAV(**vars(args))
    augmentation.run() # Run the augmentation process