import re
import os
import torch
from torch.nn.functional import ctc_loss, log_softmax
from torch.optim import RAdam
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torchmetrics.functional import word_error_rate, char_error_rate
import sentencepiece as spm
from nemo.collections.asr.modules import (
    AudioToMelSpectrogramPreprocessor,
    SpectrogramAugmentation,
    SqueezeformerEncoder,
    ConvASRDecoder,
)
from nemo.core import typecheck
from torchaudio.models.decoder import ctc_decoder

typecheck.set_typecheck_enabled(False)

# -------------------------
# Tokenizer
# -------------------------
sp = spm.SentencePieceProcessor()
sp.Load("ressources/tokenizer/128_v7.model")

# -------------------------
# CTC Decoder
# -------------------------
tokens_file = "ressources/tokenizer/128_v7.txt"
decoder = ctc_decoder(
    lexicon=None,
    tokens=tokens_file,
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
# Hyperparameters
# -------------------------
LR = 2e-4
NONE_COUNT = 0


# -------------------------
# LightningModule
# -------------------------
class MySqueezeformer(LightningModule):

    def __init__(self, LR=LR):
        super().__init__()
        self.LR = LR

        self.processor = AudioToMelSpectrogramPreprocessor(
            sample_rate=16000,
            features=80,
            n_fft=512,
            window_size=0.025,
            window_stride=0.01,
            log=True,
            frame_splicing=True,
        )
        self.augmentation = SpectrogramAugmentation(2, 5, 27, 0.05)

        self.encoder = SqueezeformerEncoder(
            feat_in=80,
            feat_out=-1,
            n_layers=16,
            d_model=144,
            adaptive_scale=True,
            time_reduce_idx=7,
            dropout_emb=0,
            dropout_att=0.1,
            subsampling_factor=4,
        )
        self.decoder = ConvASRDecoder(feat_in=144, num_classes=128)

    # -------------------------
    # Forward
    # -------------------------
    def forward(self, x, lengths):
        spec, lengths = self.processor(x, lengths)
        if self.training:
            spec = self.augmentation(spec, lengths)
        encoded = self.encoder(spec, lengths)
        decoded = self.decoder(encoded[0])

        logits_lengths = torch.tensor([len(d) for d in decoded], device=x.device)
        return decoded, logits_lengths

    # -------------------------
    # Training Step
    # -------------------------
    def training_step(self, batch, batch_idx):
        spectrograms, transcriptions, specs_lengths, transcriptions_lengths = batch
        outputs, logits_lengths = self(spectrograms, specs_lengths)
        outputs = torch.stack(outputs).transpose(0, 1)
        outputs = log_softmax(outputs, dim=2)

        loss = ctc_loss(
            outputs,
            transcriptions,
            logits_lengths,
            transcriptions_lengths,
            blank=1,
            zero_infinity=True,
        )

        global NONE_COUNT
        if torch.isnan(loss) or torch.isinf(loss):
            NONE_COUNT += 1
            self.log("N_c", float(NONE_COUNT), prog_bar=True, sync_dist=True)
            return None

        self.log("loss", loss, sync_dist=True, on_epoch=True, on_step=False)
        return loss

    # -------------------------
    # Validation Step
    # -------------------------
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        spectrograms, transcriptions, specs_lengths, transcriptions_lengths = batch
        outputs, logits_lengths = self(spectrograms, specs_lengths)

        all_transcriptions = []
        all_targets = []

        # Decode targets
        for i, tgt in enumerate(transcriptions):
            tgt_sentence = sp.Decode(tgt[: transcriptions_lengths[i]].tolist())
            all_targets.append(tgt_sentence)

        # Decode predictions
        for i, out in enumerate(outputs):
            result = decoder(out.cpu().unsqueeze(0))[0][0]
            tokens = decoder.idxs_to_tokens(result.tokens)
            transcription = "".join(tokens).replace("_", "").replace("|", "")
            transcription = " ".join(transcription.split("▁"))
            transcription = re.sub(r"-{2,}", "-", transcription)
            transcription = transcription.strip()
            all_transcriptions.append(transcription)

        wer = word_error_rate(all_transcriptions, all_targets)
        cer = char_error_rate(all_transcriptions, all_targets)

        # Compute CTC loss for logging
        stacked_outputs = torch.stack(outputs).transpose(0, 1)
        stacked_outputs = log_softmax(stacked_outputs, dim=2)
        val_loss = ctc_loss(
            stacked_outputs,
            transcriptions,
            logits_lengths,
            transcriptions_lengths,
            blank=1,
            zero_infinity=True,
        )

        self.log("val_loss", val_loss, sync_dist=True, on_epoch=True)
        self.log("wer", wer, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log("cer", cer, sync_dist=True, on_epoch=True)

    # -------------------------
    # Optimizer
    # -------------------------
    def configure_optimizers(self):
        optimizer = RAdam(
            self.parameters(),
            lr=self.LR,
            betas=[0.9, 0.98],
            weight_decay=1e-6,
            eps=1e-9,
        )
        return optimizer


# -------------------------
# Training
# -------------------------
if __name__ == "__main__":
    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(
            dirpath="./checkpoints_vZ2/val_loss",
            verbose=False,
            save_on_train_epoch_end=True,
            save_top_k=1,
            save_last=True,
            monitor="val_loss",
        ),
        ModelCheckpoint(
            dirpath="./checkpoints_vZ2/wer",
            verbose=False,
            save_on_train_epoch_end=True,
            save_top_k=1,
            save_last=False,
            monitor="wer",
        ),
        ModelCheckpoint(
            dirpath="./checkpoints_vZ2/cer",
            verbose=False,
            save_on_train_epoch_end=True,
            save_top_k=1,
            save_last=False,
            monitor="cer",
        ),
    ]

    model = MySqueezeformer()
    trainer = Trainer(
        accelerator="auto",
        precision="bf16",
        callbacks=callbacks,
        default_root_dir="./checkpoints_vZ2/logs",
        reload_dataloaders_every_n_epochs=1,
        max_epochs=300,
    )

    # trainer.fit(
    #     model, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader
    # )
