import os
import re
import numpy as np
import torch
import sentencepiece as spm
import torchaudio
import kenlm
from torchaudio.models.decoder import ctc_decoder
from torchaudio.transforms import Resample
from squeezeformer import MySqueezeformer


# -------------------------
# Paths
# -------------------------
dirname = os.path.dirname(__file__)

sp = spm.SentencePieceProcessor()
sp.Load(os.path.join(dirname, "../ressources/tokenizer/128_v7.model"))

sp_lm = spm.SentencePieceProcessor()
sp_lm.Load(os.path.join(dirname, "../ressources/tokenizer/5K.model"))

lm = kenlm.Model("./ressources/kenLM_model/kab_5k_6-gram_v2.bin")

# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Model
# -------------------------
model = MySqueezeformer().to(device)

acoustic_model_path = os.path.join(dirname, "../ressources/e2e_model/squeezeformer")

model.load_state_dict(torch.load(acoustic_model_path, map_location=device))

model.eval()

# -------------------------
# Decoder
# -------------------------
tokens_file = os.path.join(dirname, "../ressources/tokenizer/128_v7.txt")

decoder = ctc_decoder(
    tokens=tokens_file,
    lexicon=None,
    beam_size=128,
    beam_threshold=10,
    beam_size_token=10,
    nbest=50,
    log_add=True,
    blank_token="_",
    sil_token="|",
    unk_word="<unk>",
)


# -------------------------
# Inference
# -------------------------
@torch.no_grad()
def inference(audiofile: str) -> str:
    # ---- Load audio ----
    waveform, sr = torchaudio.load(audiofile)
    # Convert to 16 kHz if necessary
    target_sr = 16000
    if sr != target_sr:
        resampler = Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
        sr = target_sr
    # Convert to mono
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    waveform = waveform.to(device)

    # Shape: [B, T]
    lengths = torch.tensor([waveform.size(1)], device=device)

    # ---- Model forward ----
    outputs, _ = model.forward(waveform, lengths)

    # Move to CPU for decoder
    outputs = outputs.cpu()

    # ---- Decode ----
    results_array = decoder(outputs)[0]

    transcriptions = []
    scores = []

    for result in results_array:
        # tokens -> string
        tokens = decoder.idxs_to_tokens(result.tokens)
        transcription = "".join(tokens)

        transcription = transcription.replace("_", "")
        transcription = transcription.replace("|", "")
        transcription = transcription.replace("▁", " ")
        transcription = " ".join(transcription.split())
        transcription = re.sub(r"-{2,}", "-", transcription).strip()

        transcriptions.append(transcription)

        # ---- LM scoring ----
        lm_input = " ".join(sp_lm.Encode(transcription, out_type=str))
        lm_input = lm_input.replace("- ", "-").replace(" -", "-")

        lm_score = lm.score(lm_input)

        # weighted score
        score = lm_score * 0.25 + result.score * 0.75
        scores.append(score)

    best_idx = int(np.argmax(scores))
    return transcriptions[best_idx]
