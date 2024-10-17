# Define a class to store hyperparameters for a text-to-speech (TTS) model
class Hyperparams:
    # Seed for random number generation
    seed = 42

    # Paths for CSV metadata, audio waveforms, saved model parameters, and training logs
    csv_path = "/content/metadata.csv"
    wav_path = "/content/LJSpeech-1.1/wavs"
    save_path = "/content/result/toy_tts/params"
    log_path = "/content/result/toy_tts/train_logs"

    # Name for saving the trained model
    save_name = "SimpleTransformerTTS.pt"

    # Text transformations parameters
    symbols = [
        'EOS', ' ', '!', ',', '-', '.', ';', '?', 'a', 'b', 'c', 'd', 'e', 'f',
        'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
        'u', 'v', 'w', 'x', 'y', 'z', 'à', 'â', 'è', 'é', 'ê', 'ü', '’', '“', '”'
    ]

    # Sound transformations parameters
    sr = 22050  # Sample rate
    n_fft = 2048  # Number of points for the Fast Fourier Transform (FFT)
    n_stft = int((n_fft // 2) + 1)  # Number of Short-Time Fourier Transform (STFT) components

    frame_shift = 0.0125  # Time in seconds between consecutive frames
    hop_length = int(n_fft / 8.0)  # Hop length for STFT

    frame_length = 0.05  # Length of each frame in seconds
    win_length = int(n_fft / 2.0)  # Window length for STFT

    mel_freq = 128  # Number of mel frequency bins
    max_mel_time = 1024  # Maximum mel spectrogram time steps

    max_db = 100  # Maximum decibel value
    scale_db = 10  # Scaling factor for decibels
    ref = 4.0  # Reference value for spectrogram scaling
    power = 2.0  # Exponent for spectrogram scaling
    norm_db = 10  # Normalization factor for decibels

    ampl_multiplier = 10.0  # Multiplier for amplitude
    ampl_amin = 1e-10  # Minimum amplitude
    db_multiplier = 1.0  # Multiplier for decibels
    ampl_ref = 1.0  # Amplitude reference value
    ampl_power = 1.0  # Exponent for amplitude scaling

    # Model parameters
    text_num_embeddings = 2 * len(symbols)  # Number of embeddings for text symbols
    embedding_size = 256  # Size of the text embeddings
    encoder_embedding_size = 512  # Size of the encoder embeddings

    dim_feedforward = 1024  # Feedforward dimension in the model
    postnet_embedding_size = 1024  # Size of the embeddings in the postnet

    encoder_kernel_size = 3  # Kernel size for the encoder
    postnet_kernel_size = 5  # Kernel size for the postnet

    # Other parameters
    batch_size = 32  # Size of the training batch
    grad_clip = 1.0  # Gradient clipping threshold
    lr = 2.0 * 1e-4  # Learning rate
    r_gate = 1.0  # Ratio of gate loss to the mel loss

    step_print = 1000  # Interval for printing training information
    step_test = 8000  # Interval for testing the model
    step_save = 8000  # Interval for saving the model


# Create an instance of the Hyperparams class
hp = Hyperparams()

# Print the symbols and the number of symbols
if __name__ == "__main__":
    print(hp.symbols)
    print(len(hp.symbols))
