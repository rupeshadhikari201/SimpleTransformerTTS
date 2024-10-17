import numpy as np
import pydub
from hyperparams import hp

def write_mp3(
  x, 
  f="audio.mp3", 
  sr=hp.sr, 
  normalized=True
):
  
  """
    Convert a numpy array to an MP3 file.

    Parameters:
    - x: numpy array representing the audio waveform
    - f: output file path for the MP3 file (default: "audio.mp3")
    - sr: sample rate of the audio waveform (default: hp.sr from hyperparameters)
    - normalized: whether the input array is normalized (default: True)

    Note:
    - If normalized is True, each item in the array should be a float in the range [-1, 1).
    - If normalized is False, the array is treated as raw audio samples.

    Uses the Pydub library to handle the conversion and export.

    Returns:
    - None
    """
  
   # Determine the number of audio channels
  channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
  
   # Convert the array to int16 format for Pydub
  if normalized:  # normalized array - each item should be a float in [-1, 1)
      y = np.int16(x * 2 ** 15)
  else:
      y = np.int16(x)
      
  # Create an AudioSegment from the numpy array
  song = pydub.AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
  
  # Export the AudioSegment to an MP3 file
  song.export(f, format="mp3", bitrate="320k")