# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from argparse import ArgumentParser
from typing import Optional, Tuple
import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim
import math

from .transformer import (
    Transformer, generate_position_encoding,
    generate_key_padding_mask, generate_square_subsequent_mask)
from .asr import InvertedResidual

ACTION_BLANK = 0
ACTION_STAY = 1
ACTION_STEP = 2

@torch.no_grad()
def generate_padding_mask(x: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: tensor of shape [batch_size, length]
        length: tensor of shape [batch_size]
    Returns:
        float tensor of shape [batch_size, length]
    """
    assert x.dim() == 2
    assert length.dim() == 1
    return (torch.arange(x.shape[1], device=x.device)[None, :] < length[:, None]).float()

class VoiceDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size=256) -> None:
        super().__init__()
        half_hidden_size = hidden_size // 2
        self.layers = nn.Sequential(
            InvertedResidual(in_channels, hidden_size, kernel_size=65, use_residual=False),
            InvertedResidual(hidden_size, hidden_size, kernel_size=65),
            nn.ConvTranspose1d(hidden_size, half_hidden_size, kernel_size=5, padding=2, stride=2),
            InvertedResidual(half_hidden_size, half_hidden_size, kernel_size=17),
            InvertedResidual(half_hidden_size, half_hidden_size, kernel_size=11),
            nn.Conv1d(half_hidden_size, out_channels, kernel_size=1, bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class CustomSchedule(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super(CustomSchedule, self).__init__(optimizer)

    def get_lr(self):
        step = max(1, self._step_count) / 3
        arg1 = 1 / math.sqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        x = min(arg1, arg2) / math.sqrt(self.d_model)
        return [base_lr * x
                for base_lr in self.base_lrs]

def adjust_size(
    x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.shape[1] > y.shape[1]:
        return x[:, :y.shape[1]], y
    if x.shape[1] < y.shape[1]:
        return x, y[:, :x.shape[1]]
    return x, y

class WORLDNorm(nn.Module):
    def __init__(self, logspc_size: int, codeap_size: int, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.f0_std = nn.Parameter(
            torch.ones([1], **factory_kwargs),
            requires_grad=False)
        self.f0_mean = nn.Parameter(
            torch.zeros([1], **factory_kwargs),
            requires_grad=False)
        self.logspc_std = nn.Parameter(
            torch.ones([logspc_size], **factory_kwargs),
            requires_grad=False)
        self.logspc_mean = nn.Parameter(
            torch.zeros([logspc_size], **factory_kwargs),
            requires_grad=False)
        self.codeap_std = nn.Parameter(
            torch.ones([codeap_size], **factory_kwargs),
            requires_grad=False)
        self.codeap_mean = nn.Parameter(
            torch.zeros([codeap_size], **factory_kwargs),
            requires_grad=False)

    def forward(self, f0, mcep, codeap):
        return self.normalize(f0, mcep, codeap)

    @torch.no_grad()
    def normalize(
        self, f0: torch.Tensor, mcep: torch.Tensor, codeap: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f0 = (f0 - self.f0_mean) / self.f0_std
        mcep = (mcep - self.logspc_mean) / self.logspc_std
        codeap = (codeap - self.codeap_mean) / self.codeap_std
        return f0, mcep, codeap

    @torch.no_grad()
    def unnormalize(
        self, f0: torch.Tensor, mcep: torch.Tensor, codeap: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f0 = self.f0_std * f0 + self.f0_mean 
        mcep = self.logspc_std * mcep + self.logspc_mean
        codeap = self.codeap_std * codeap + self.codeap_mean
        return f0, mcep, codeap

class WORLDLoss(nn.Module):
    def __init__(self, sample_rate: int = 16000, n_fft: int = 512, device=None, dtype=None):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.l1_loss = nn.L1Loss(reduction='none')

        f = (sample_rate / n_fft) * torch.arange(
            n_fft // 2 + 1, device=device, dtype=dtype if dtype is not None else torch.float32)
        dm = 1127 / (700 + f)
        logspc_weights = dm / torch.sum(dm)
        self.register_buffer('logspc_weights', logspc_weights, persistent=False)

    def forward(
        self, length: torch.Tensor,
        hasf0_hat: torch.Tensor, f0_hat: torch.Tensor, logspc_hat: torch.Tensor, codeap_hat: torch.Tensor,
        hasf0: torch.Tensor, f0: torch.Tensor, logspc: torch.Tensor, codeap: torch.Tensor
        ) -> torch.Tensor:

        hasf0_hat, hasf0 = adjust_size(hasf0_hat, hasf0)
        f0_hat, f0 = adjust_size(f0_hat, f0)
        logspc_hat, logspc = adjust_size(logspc_hat, logspc)
        codeap_hat, codeap = adjust_size(codeap_hat, codeap)

        mask = generate_padding_mask(f0, length)
        hasf0_loss = self.bce_loss(hasf0_hat, hasf0) * mask
        f0_loss = self.l1_loss(f0_hat, f0) * hasf0 * mask
        logspc_loss = torch.sum(self.l1_loss(logspc_hat, logspc) * self.logspc_weights[None, None, :], axis=2) * mask
        codeap_loss = torch.mean(self.l1_loss(codeap_hat, codeap), axis=2) * mask
        mask_sum = torch.sum(mask)
        hasf0_loss = torch.sum(hasf0_loss) / mask_sum
        f0_loss = torch.sum(f0_loss) / mask_sum
        logspc_loss = torch.sum(logspc_loss) / mask_sum
        codeap_loss = torch.sum(codeap_loss) / mask_sum
        return hasf0_loss, f0_loss, logspc_loss, codeap_loss

class CharToAudioModel(pl.LightningModule):
    def __init__(
        self, vocab_size: int, hidden_size: int, filter_size: int,
        num_layers: int, num_headers: int, learning_rate: float) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_actions = 3
        self.sample_rate = 16000
        self.n_fft = 512
        self.hasf0_size = 1
        self.f0_size = 1
        self.logspc_size = self.n_fft // 2 + 1
        self.codeap_size = 1
        self.transformer = Transformer(hidden_size, filter_size, num_layers, num_headers)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(0.1, inplace=True)
        self.out_proj = nn.Linear(hidden_size, self.num_actions)
        self.audio_size = self.hasf0_size + self.f0_size + self.logspc_size + self.codeap_size
        self.world_out_proj = VoiceDecoder(hidden_size, self.audio_size)
        self.criteria = nn.CrossEntropyLoss(reduction='none')
        self.world_norm = WORLDNorm(self.logspc_size, self.codeap_size)
        self.world_criteria = WORLDLoss(sample_rate=self.sample_rate, n_fft=self.n_fft)
    
    def forward(self, src_ids, src_ids_len, tgt_in_ids):
        src = self.embedding(src_ids) * self.hidden_size ** 0.5
        tgt_in = self.embedding(tgt_in_ids) * self.hidden_size ** 0.5
        src_pos_encoding = generate_position_encoding(src)
        tgt_pos_encoding = generate_position_encoding(tgt_in)
        src = self.dropout(src + src_pos_encoding)
        tgt_in = self.dropout(tgt_in + tgt_pos_encoding)
        src_key_padding_mask = generate_key_padding_mask(src, src_ids_len)

        tgt_mask = generate_square_subsequent_mask(
            tgt_in.shape[1], device=tgt_in.device)
        #print(memory_key_padding_mask.shape, decoder_self_attention_bias.shape)

        decoder_outputs = self.transformer(src, tgt_in, src_key_padding_mask=src_key_padding_mask, tgt_mask=tgt_mask)
        batch_size = -1 # torch.shape(inputs)[0]
        length = decoder_outputs.shape[1]
        decoder_outputs = torch.reshape(decoder_outputs, [batch_size, length, self.hidden_size])
        # decoder_outputs: [batch_size, target_len, hidden_size]

        logits = self.out_proj(decoder_outputs)
        # logits: [batch_size, target_len, vocab_size]

        decoder_outputs_trans = torch.transpose(decoder_outputs, 1, 2)
        world_out_trans = self.world_out_proj(decoder_outputs_trans)
        world_out = torch.transpose(world_out_trans, 1, 2)
        # world_out: [batch_size, target_len, audio_size]

        hasf0_hat, f0_hat, logspc_hat, codeap_hat = torch.split(world_out, [
            self.hasf0_size,
            self.f0_size,
            self.logspc_size,
            self.codeap_size
        ], dim=2)
        hasf0_hat = hasf0_hat[:, :, 0]
        f0_hat = f0_hat[:, :, 0]
        return logits, hasf0_hat, f0_hat, logspc_hat, codeap_hat

    def _calc_batch_loss(self, batch):
        (f0, f0_len, logspc, codeap), (text, text_len), (aligntext, aligntext_len) = batch
        hasf0 = (f0 >= 30.0).to(torch.float32)
        f0, logspc, codeap = self.world_norm.normalize(f0, logspc, codeap)

        src_ids, src_ids_len = self._create_source_input(text, text_len)
        tgt_in_ids = self._create_target_input(aligntext)
        tgt_out_ids = self._generate_action(aligntext)
        tgt_out_ids_len = aligntext_len

        logits, hasf0_hat, f0_hat, logspc_hat, codeap_hat = self.forward(src_ids, src_ids_len, tgt_in_ids)
        logits = torch.transpose(logits, 1, 2)

        hasf0_loss, f0_loss, logspc_loss, codeap_loss = self.world_criteria(
            f0_len, hasf0_hat, f0_hat, logspc_hat, codeap_hat, hasf0, f0, logspc, codeap)
        align_loss = self._calc_align_loss(logits, tgt_out_ids, tgt_out_ids_len)

        return align_loss, hasf0_loss, f0_loss, logspc_loss, codeap_loss

    def predict(self, src_ids: torch.Tensor, src_ids_len, max_steps=100):
        src = self.embedding(src_ids) * self.hidden_size ** 0.5
        src_pos_encoding = generate_position_encoding(src)
        src = self.dropout(src + src_pos_encoding)
        src_key_padding_mask = generate_key_padding_mask(src, src_ids_len)
        memory = self.transformer.encode(src, src_key_padding_mask)

        tgt_in_ids = torch.zeros([src_ids.shape[0], 1], dtype=src_ids.dtype, device=src_ids.device)
        tgt_pos_encoding = generate_position_encoding(
            torch.zeros(
                [1, max_steps, self.hidden_size],
                device=src.device, dtype=src.dtype))

        cache = {}
        dec_out = []
        tgt_out = []
        src_pos = torch.zeros_like(src[:, :1], dtype=torch.long)

        for pos in range(max_steps):
            tgt_in = self.embedding(tgt_in_ids) * self.hidden_size ** 0.5
            tgt_in = self.dropout(tgt_in + tgt_pos_encoding[None, pos:pos+1, :])
            dec = self.transformer.decode(
                tgt_in, memory, tgt_mask=None,
                memory_key_padding_mask=src_key_padding_mask, cache=cache)
            # dec: [batch_size, alightext_len, hidden_size]
            logits = self.out_proj(dec[:, -1:, :])
            preds = logits.argmax(axis=-1)
            src_pos = torch.where(
                preds == ACTION_BLANK,
                torch.floor_divide(src_pos + 1, 2) * 2,
                torch.where(
                    preds == ACTION_STEP,
                    src_pos + 1,
                    src_pos))
            tgt_out_id = torch.where(
                src_pos % 2 == 1,
                src_ids[torch.floor_divide(src_pos, 2)][:, None],
                0
            )
            tgt_in_ids = tgt_out_id

            dec_out.append(dec)
            tgt_out.append(tgt_out_id)

        dec_out = torch.cat(dec_out, axis=1)
        tgt_out = torch.cat(tgt_out, axis=1)

        dec_out_trans = torch.transpose(dec_out, 1, 2)
        world_out_trans = self.world_out_proj(dec_out_trans)
        world_out = torch.transpose(world_out_trans, 1, 2)
        # world_out: [batch_size, target_len, audio_size]

        hasf0_hat, f0_hat, logspc_hat, codeap_hat = torch.split(world_out, [
            self.hasf0_size,
            self.f0_size,
            self.logspc_size,
            self.codeap_size
        ], dim=2)
        hasf0_hat = hasf0_hat[:, :, 0]
        f0_hat = f0_hat[:, :, 0]

        return tgt_out, hasf0_hat, f0_hat, logspc_hat, codeap_hat

    def _create_source_input(self, text, text_len):
        source_input = torch.cat([
            text,
            torch.zeros_like(text[:, :1])
            ], axis=1)
        source_input_length = text_len + 1
        return source_input, source_input_length

    def _create_target_input(self, aligntext):
        return torch.cat([
            torch.zeros_like(aligntext[:, -1:]),
            aligntext[:, :-1]
            ], axis=1)

    def _generate_action(self, aligntext):
        prev_aligntext = torch.cat([
            torch.zeros_like(aligntext[:, -1:]),
            aligntext[:, :-1]],
            axis=1)
        return torch.where(
            aligntext == 0,
            ACTION_BLANK,
            torch.where(
                prev_aligntext == aligntext,
                ACTION_STAY, ACTION_STEP))

    def _calc_align_loss(self, logits, aligntext, aligntext_len):
        aligntext_mask = generate_padding_mask(aligntext, aligntext_len)
        align_loss = self.criteria(logits, aligntext)
        align_loss = torch.sum(align_loss * aligntext_mask) / torch.sum(aligntext_mask)
        return align_loss

    def training_step(self, batch, batch_idx):
        align_loss, hasf0_loss, f0_loss, logspc_loss, codeap_loss = self._calc_batch_loss(batch)
        loss = align_loss + hasf0_loss + f0_loss + logspc_loss + codeap_loss

        self.log('train_align_loss', align_loss)
        self.log('train_hasf0_loss', hasf0_loss)
        self.log('train_f0_loss', f0_loss)
        self.log('train_logspc_loss', logspc_loss)
        self.log('train_codeap_loss', codeap_loss)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        align_loss, hasf0_loss, f0_loss, logspc_loss, codeap_loss = self._calc_batch_loss(batch)
        loss = align_loss + hasf0_loss + f0_loss + logspc_loss + codeap_loss
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        align_loss, hasf0_loss, f0_loss, logspc_loss, codeap_loss = self._calc_batch_loss(batch)
        loss = align_loss + hasf0_loss + f0_loss + logspc_loss + codeap_loss
        self.log('test_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9)#,
            #weight_decay=0.0001)
        scheduler = CustomSchedule(optimizer, d_model=self.hparams.hidden_size)
        lr_dict = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
            "name": "CustomSchedule",
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_dict}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hidden_size', type=int, default=128)
        parser.add_argument('--filter_size', type=int, default=512)
        parser.add_argument('--num_layers', type=int, default=4)
        parser.add_argument('--num_headers', type=int, default=8)
        parser.add_argument('--learning_rate', type=float, default=1.0)
        parser.add_argument('--audio_stat', type=str, default='data/stat_ljspeech.pt')
        return parser

    @staticmethod
    def from_argparse_args(args):
        model = CharToAudioModel(
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            filter_size=args.filter_size,
            num_layers=args.num_layers,
            num_headers=args.num_headers,
            learning_rate=args.learning_rate)
        model.world_norm.load_state_dict(torch.load(args.audio_stat))
        return model