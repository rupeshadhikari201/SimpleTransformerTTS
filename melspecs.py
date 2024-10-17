from hyperparams import hp
import torch
import torchaudio
from torchaudio.functional import spectrogram


# Spectrogram transformation
'''
This function computes the spectrogram of an audio waveform. 
It converts the raw audio signal into a time-frequency representation using the Short-Time Fourier Transform (STFT)
'''
spec_transform = torchaudio.transforms.Spectrogram(
    n_fft=hp.n_fft,             # Size of the FFT window.
    win_length=hp.win_length,   # Window size for each frame.
    hop_length=hp.hop_length,   # Number of samples between successive frames.
    power=hp.power              # Exponent for the magnitude spectrogram.
)

# Mel scale transformation
#  Applies a Mel filterbank to the spectrogram to obtain a Mel spectrogram.
mel_scale_transform = torchaudio.transforms.MelScale(
  n_mels=hp.mel_freq,           # Number of Mel filterbanks.
  sample_rate=hp.sr,            # Sample rate of the audio.
  n_stft=hp.n_stft              # Number of Fourier bins.
)


# Inverse Mel scale transformation
# Computes the inverse of the Mel scale transformation. 
# It converts a Mel spectrogram back to a linear-scale spectrogram
mel_inverse_transform = torchaudio.transforms.InverseMelScale(
  n_mels=hp.mel_freq, 
  sample_rate=hp.sr, 
  n_stft=hp.n_stft
)

# Griffin-Lim transformation
# Performs the Griffin-Lim algorithm, which is an iterative method for reconstructing a time-domain signal from its magnitude spectrogram.
griffnlim_transform = torchaudio.transforms.GriffinLim(
    n_fft=hp.n_fft,               # Size of the FFT window.
    win_length=hp.win_length,     # Window size for each frame.
    hop_length=hp.hop_length      # Number of samples between successive frames.  
)


# Normalizes a Mel spectrogram in decibels to a specified range.
def norm_mel_spec_db(mel_spec):  
  mel_spec = ((2.0*mel_spec - hp.min_level_db) / (hp.max_db/hp.norm_db)) - 1.0
  mel_spec = torch.clip(mel_spec, -hp.ref*hp.norm_db, hp.ref*hp.norm_db)
  return mel_spec

# Denormalizes a Mel spectrogram back to its original range.
def denorm_mel_spec_db(mel_spec):
  mel_spec = (((1.0 + mel_spec) * (hp.max_db/hp.norm_db)) + hp.min_level_db) / 2.0 
  return mel_spec

#  Converts a power spectrogram to decibels with scaling and clipping.
def pow_to_db_mel_spec(mel_spec):
  mel_spec = torchaudio.functional.amplitude_to_DB(
    mel_spec,
    multiplier = hp.ampl_multiplier, 
    amin = hp.ampl_amin, 
    db_multiplier = hp.db_multiplier, 
    top_db = hp.max_db
  )
  mel_spec = mel_spec/hp.scale_db
  return mel_spec


# Converts a decibel spectrogram back to power.
def db_to_power_mel_spec(mel_spec):
  mel_spec = mel_spec*hp.scale_db
  mel_spec = torchaudio.functional.DB_to_amplitude(
    mel_spec,
    ref=hp.ampl_ref,
    power=hp.ampl_power
  )  
  return mel_spec


# Takes an audio waveform, computes its Mel spectrogram, and converts it to a normalized decibel scale for further processing.
def convert_to_mel_spec(wav):
  spec = spec_transform(wav)
  mel_spec = mel_scale_transform(spec)
  db_mel_spec = pow_to_db_mel_spec(mel_spec)
  db_mel_spec = db_mel_spec.squeeze(0)
  return db_mel_spec


#  Takes a normalized decibel Mel spectrogram, converts it back to the power spectrogram, and reconstructs an audio waveform using the Griffin-Lim 
def inverse_mel_spec_to_wav(mel_spec):
  power_mel_spec = db_to_power_mel_spec(mel_spec)
  spectrogram = mel_inverse_transform(power_mel_spec)
  pseudo_wav = griffnlim_transform(spectrogram)
  return pseudo_wav


'''
Loads an audio waveform from a specified file.
Applies the convert_to_mel_spec function to obtain a Mel spectrogram.
Prints the shape of the Mel spectrogram.
Applies the inverse_mel_spec_to_wav function to reconstruct a pseudo waveform.
Prints the shape of the pseudo waveform.
'''
if __name__ == "__main__":
  wav_path = f"{hp.wav_path}/LJ023-0073.wav" 
  waveform, sample_rate = torchaudio.load(wav_path, normalize=True)
  mel_spec = convert_to_mel_spec(waveform)
  print("mel_spec:", mel_spec.shape)

  pseudo_wav = inverse_mel_spec_to_wav(mel_spec.cuda())
  print("pseudo_wav:", pseudo_wav.shape)
  
  
'''
The overall purpose of this code seems to be to demonstrate a pipeline for converting an audio waveform into a Mel spectrogram, performing some transformations, and then reconstructing a pseudo waveform from the modified Mel spectrogram. This type of processing is often used in tasks related to audio signal processing and speech recognition. The code seems to be set up with hyperparameters (hp) for controlling various aspects of the transformation.
'''