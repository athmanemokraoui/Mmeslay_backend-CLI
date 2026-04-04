import torch
import numpy as np
import re
import sentencepiece as spm
import kenlm

from data_loading import test_dataloader
from squeezeformer import MySqueezeformer
from torchmetrics.functional import word_error_rate, char_error_rate
from torchaudio.models.decoder import ctc_decoder

# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Tokenizers & LM
# -------------------------
sp = spm.SentencePieceProcessor()
sp.Load("ressources/tokenizer/128_v7.model")

sp_lm = spm.SentencePieceProcessor()
sp_lm.Load("ressources/tokenizer/5K.model")

lm = kenlm.Model("./ressources/kenLM_model/kab_5k_trigram.bin")

# -------------------------
# Model
# -------------------------
MODEL = MySqueezeformer().to(device)

MODEL.load_state_dict(
    torch.load("ressources/e2e_model/squeezeformer", map_location=device), strict=False
)

MODEL.eval()

# -------------------------
# Decoder
# -------------------------
decoder = ctc_decoder(
    tokens="ressources/tokenizer/128_v7.txt",
    lexicon=None,
    beam_size=1,
    beam_threshold=1,
    beam_size_token=1,
    nbest=1,
    log_add=True,
    blank_token="_",
    sil_token="|",
    unk_word="<unk>",
)


# -------------------------
# Helpers
# -------------------------
def clean_text(tokens):
    text = "".join(tokens)
    text = text.replace("_", "")
    text = text.replace("|", "")
    text = text.replace("▁", " ")
    text = " ".join(text.split())
    text = re.sub(r"-{2,}", "-", text)
    return text.strip()


@torch.no_grad()
def evaluate():
    all_transcriptions = []
    all_targets = []

    for batch in test_dataloader:
        if batch is None:
            continue

        inputs, targets, input_lengths, target_lengths = batch

        inputs = inputs.to(device)
        input_lengths = input_lengths.to(device)

        # ---- Forward ----
        outputs, _ = MODEL.forward(inputs, input_lengths)

        # decoder expects CPU
        outputs = outputs.cpu()

        # ---- Decode batch directly (faster) ----
        batch_results = decoder(outputs)

        # ---- Targets ----
        for i in range(len(targets)):
            tgt = targets[i][: target_lengths[i]].tolist()
            target_sentence = sp.Decode(tgt)
            all_targets.append(target_sentence)

        # ---- Predictions ----
        for results_array in batch_results:
            transcriptions = []
            scores = []

            for result in results_array:
                tokens = decoder.idxs_to_tokens(result.tokens)

                transcription = clean_text(tokens)
                transcriptions.append(transcription)

                # ---- LM scoring ----
                lm_input = " ".join(sp_lm.Encode(transcription, out_type=str))
                lm_input = lm_input.replace("- ", "-").replace(" -", "-")

                lm_score = lm.score(lm_input)
                score = lm_score * 0.25 + result.score * 0.75
                scores.append(score)

            best_idx = int(np.argmax(scores))
            best_transcription = transcriptions[best_idx]

            print(best_transcription)
            all_transcriptions.append(best_transcription)

    # -------------------------
    # Metrics
    # -------------------------
    wer = word_error_rate(all_transcriptions, all_targets)
    cer = char_error_rate(all_transcriptions, all_targets)

    print(f"Average Word Error Rate: {wer * 100:.2f}%")
    print(f"Average Character Error Rate: {cer * 100:.2f}%")


if __name__ == "__main__":
    evaluate()
