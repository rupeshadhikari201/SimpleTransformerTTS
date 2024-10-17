# Import necessary libraries and modules
import torch
import torchaudio
import pandas as pd

# Import hyperparameters, text_to_seq, and other utility functions
from hyperparams import hp
from text_to_seq import text_to_seq
from mask_from_seq_lengths import mask_from_seq_lengths
from melspecs import convert_to_mel_spec

# Define a custom dataset class for the text-to-mel spectrogram task
class TextMelDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        # Initialize the dataset with a dataframe and an empty cache for caching processed data
        self.df = df
        self.cache = {}

    # Function to process a single row and return text and corresponding mel spectrogram
    def get_item(self, row):
        wav_id = row["wav"]                  
        wav_path = f"{hp.wav_path}/{wav_id}.wav"

        # Convert normalized text to sequence
        text = row["text_norm"]
        text = text_to_seq(text)

        # Load audio waveform and ensure the sample rate matches the specified hyperparameter
        waveform, sample_rate = torchaudio.load(wav_path, normalize=True)
        assert sample_rate == hp.sr

        # Convert the waveform to mel spectrogram
        mel = convert_to_mel_spec(waveform)

        return (text, mel)

    # Function to get an item from the dataset given an index
    def __getitem__(self, index):
        row = self.df.iloc[index]
        wav_id = row["wav"]

        # Check if the processed data is already in the cache, if not, process and cache it
        text_mel = self.cache.get(wav_id)
        if text_mel is None:
            text_mel = self.get_item(row)
            self.cache[wav_id] = text_mel

        return text_mel

    # Function to get the length of the dataset
    def __len__(self):
        return len(self.df)


# Define a collate function to process and pad batches of text and mel spectrograms
def text_mel_collate_fn(batch):
    # Calculate the maximum lengths of text and mel spectrogram in the batch
    text_length_max = torch.tensor([text.shape[-1] for text, _ in batch], dtype=torch.int32).max()
    mel_length_max = torch.tensor([mel.shape[-1] for _, mel in batch], dtype=torch.int32).max()

    # Initialize lists to store lengths and padded sequences
    text_lengths = []
    mel_lengths = []
    texts_padded = []
    mels_padded = []

    # Iterate over the batch to pad text and mel sequences
    for text, mel in batch:
        text_length = text.shape[-1]      
        
        # Pad text sequences with zeros
        text_padded = torch.nn.functional.pad(
            text,
            pad=[0, text_length_max - text_length],
            value=0
        )

        mel_length = mel.shape[-1]
        
        # Pad mel spectrogram sequences with zeros
        mel_padded = torch.nn.functional.pad(
            mel,
            pad=[0, mel_length_max - mel_length],
            value=0
        )

        # Append processed data to respective lists
        text_lengths.append(text_length)
        mel_lengths.append(mel_length)
        texts_padded.append(text_padded)
        mels_padded.append(mel_padded)

    # Convert lists to tensors and transpose mel spectrograms
    text_lengths = torch.tensor(text_lengths, dtype=torch.int32)
    mel_lengths = torch.tensor(mel_lengths, dtype=torch.int32)
    texts_padded = torch.stack(texts_padded, 0)
    mels_padded = torch.stack(mels_padded, 0).transpose(1, 2)

    # Create a mask for the stop tokens based on mel spectrogram lengths
    stop_token_padded = mask_from_seq_lengths(
        mel_lengths,
        mel_length_max
    )
    # Invert the mask and set the last element to 1.0
    stop_token_padded = (~stop_token_padded).float()
    stop_token_padded[:, -1] = 1.0

    # Return the processed data as a tuple
    return texts_padded, text_lengths, mels_padded, mel_lengths, stop_token_padded


if __name__ == "__main__":
    # Read the CSV file containing information about the dataset
    df = pd.read_csv(hp.csv_path)
    # Create an instance of the TextMelDataset
    dataset = TextMelDataset(df)

    # Create a data loader for training using the defined collate function
    train_loader = torch.utils.data.DataLoader(
        dataset, 
        num_workers=2, 
        shuffle=True,
        sampler=None, 
        batch_size=hp.batch_size,
        pin_memory=True, 
        drop_last=True, 
        collate_fn=text_mel_collate_fn
    )
    
    # Function to create a string representing the shape of tensors
    def names_shape(names, shape):  
        assert len(names) == len(shape)
        return "(" + ", ".join([f"{k}={v}" for k, v in list(zip(names, shape))]) + ")"

    # Iterate over the first batch and print the shapes of processed data
    for i, batch in enumerate(train_loader):
        text_padded, text_lengths, mel_padded, mel_lengths, stop_token_padded = batch

        # Print information about the shapes of tensors in the batch
        print(f"=========batch {i}=========")
        print("text_padded:", names_shape(["N", "S"], text_padded.shape))
        print("text_lengths:", names_shape(["N"], text_lengths.shape))
        print("mel_padded:", names_shape(["N", "TIME", "FREQ"], mel_padded.shape))
        print("mel_lengths:", names_shape(["N"], mel_lengths.shape))
        print("stop_token_padded:", names_shape(["N", "TIME"], stop_token_padded.shape))

        # Stop after printing the first batch
        if i > 0:
            break

# Summary:
# The provided script defines a PyTorch dataset and associated collate function for a text-to-mel spectrogram task.
# It loads audio waveforms and corresponding normalized text from a dataframe, converts text to sequences, 
# converts audio waveforms to mel spectrograms, and pads sequences to form batches.
# The collate function handles batch processing, ensuring that sequences are padded appropriately.
# The script then demonstrates the use of the dataset and data loader, printing the shapes of tensors in the first batch.
