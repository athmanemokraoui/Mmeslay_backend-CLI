import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import sentencepiece as spm


import torchaudio
from torchaudio.transforms import Resample

# -------------------------
# Tokenizer
# -------------------------
sp = spm.SentencePieceProcessor()
sp.Load("./ressources/tokenizer/128_v7.model")

# -------------------------
# Load CSVs
# -------------------------
train_data = pd.read_csv("./ressources/train.csv", low_memory=False)
validation_data = pd.read_csv("./ressources/dev.csv", low_memory=False)
test_data = pd.read_csv("./ressources/test.csv", low_memory=False)

X_train, y_train = train_data["path"], train_data["sentence"]
X_val, y_val = validation_data["path"], validation_data["sentence"]
X_test, y_test = test_data["path"], test_data["sentence"]

del train_data, validation_data, test_data


audio_location = os.environ.get("AUDIO_LOCATION")


# -------------------------
# Collate Function
# -------------------------
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    transcriptions, waveforms, audio_lengths = zip(*batch)

    transcriptions = [torch.tensor(t, dtype=torch.long) for t in transcriptions]
    waveforms = [torch.tensor(w, dtype=torch.float32) for w in waveforms]

    transcription_lengths = torch.tensor(
        [t.size(0) for t in transcriptions], dtype=torch.int32
    )
    audio_lengths = torch.tensor(audio_lengths, dtype=torch.int32)

    padded_waveforms = pad_sequence(waveforms, batch_first=True, padding_value=0.0)
    padded_transcriptions = pad_sequence(
        transcriptions, batch_first=True, padding_value=0
    )

    return padded_waveforms, padded_transcriptions, audio_lengths, transcription_lengths


# -------------------------
# Dataset
# -------------------------
class AudioDataset(Dataset):
    def __init__(self, X, y, audio_location=audio_location, train=False):
        self.audio_dirs = X.reset_index(drop=True)
        self.transcriptions = y.reset_index(drop=True)
        self.train = train
        self.audio_location = audio_location

        self.target_sr = 16000
        self.resampler = None

    def __len__(self):
        return len(self.transcriptions)

    def __getitem__(self, idx):
        paths = str(self.audio_dirs[idx]).split(",")

        if self.train:
            chosen = random.randint(0, len(paths) - 1)
        else:
            chosen = 0

        audio_location = f"{self.audio_location}/{paths[chosen]}.mp3"

        # ---- Text ----
        transcription = sp.Encode(self.transcriptions[idx], out_type=int)

        # ---- Audio ----
        waveform, sr = torchaudio.load(audio_location)

        # Convert to mono
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.target_sr:
            if self.resampler is None or self.resampler.orig_freq != sr:
                self.resampler = Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = self.resampler(waveform)

        waveform = waveform.squeeze(0)  # [T]

        return transcription, waveform, waveform.size(0)


# -------------------------
# Datasets
# -------------------------
train_data = AudioDataset(X_train, y_train, train=True)
validation_data = AudioDataset(X_val, y_val)
test_data = AudioDataset(X_test, y_test)

# -------------------------
# DataLoaders
# -------------------------
train_dataloader = DataLoader(
    train_data,
    shuffle=True,
    drop_last=True,
    batch_size=64,
    num_workers=8,
    collate_fn=collate_fn,
    pin_memory=True,
    persistent_workers=True,
)

validation_dataloader = DataLoader(
    validation_data,
    batch_size=64,
    num_workers=4,
    collate_fn=collate_fn,
    persistent_workers=True,
)

test_dataloader = DataLoader(
    test_data,
    batch_size=4,
    num_workers=4,
    collate_fn=collate_fn,
)
