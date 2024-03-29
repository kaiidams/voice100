# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from typing import Text
from argparse import ArgumentParser
import torch
from tqdm import tqdm

from .data_modules import AudioTextDataModule


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
    return (torch.arange(x.shape[1], device=x.device)[None, :] < length[:, None]).to(x.dtype)


def calc_stat(data: AudioTextDataModule, output_path: Text):

    f0_dim, logspc_or_mcep_size, codeap_size = data.audio_transform.vocoder.output_dims
    print(logspc_or_mcep_size)
    assert f0_dim == 1
    assert logspc_or_mcep_size == 257 or logspc_or_mcep_size == 25
    assert codeap_size == 1

    f0_sum = torch.zeros(1, dtype=torch.double)
    logspc_sum = torch.zeros(logspc_or_mcep_size, dtype=torch.double)
    codeap_sum = torch.zeros(codeap_size, dtype=torch.double)
    f0_sqrsum = torch.zeros(1, dtype=torch.double)
    logspc_sqrsum = torch.zeros(logspc_or_mcep_size, dtype=torch.double)
    codeap_sqrsum = torch.zeros(codeap_size, dtype=torch.double)
    f0_count = 0
    logspc_count = 0
    for batch_idx, batch in enumerate(tqdm(data.predict_dataloader())):
        (f0, f0_len, logspc, codeap), (text, text_len) = batch
        with torch.no_grad():
            mask = generate_padding_mask(f0, f0_len)
            f0mask = (f0 > 30.0).float() * mask
            codeapmask = (codeap < -0.2).float() * mask[:, :, None]

            f0_sum += torch.sum(f0 * f0mask)
            f0_sqrsum += torch.sum(f0 ** 2 * f0mask)
            f0_count += torch.sum(f0mask)

            logspc_sum += torch.sum(torch.sum(logspc * mask[:, :, None], axis=1), axis=0)
            logspc_sqrsum += torch.sum(torch.sum(logspc ** 2 * mask[:, :, None], axis=1), axis=0)
            logspc_count += torch.sum(mask)

            codeap_sum += torch.sum(torch.sum(codeap * codeapmask, axis=1), axis=0)
            codeap_sqrsum += torch.sum(torch.sum(codeap ** 2 * codeapmask, axis=1), axis=0)

    codeap_count = logspc_count
    state_dict = {
        'f0_mean': f0_sum / f0_count,
        'f0_std': torch.sqrt((f0_sqrsum / f0_count) - (f0_sum / f0_count) ** 2),
        'logspc_mean': logspc_sum / logspc_count,
        'logspc_std': torch.sqrt((logspc_sqrsum / logspc_count) - (logspc_sum / logspc_count) ** 2),
        'codeap_mean': codeap_sum / codeap_count,
        'codeap_std': torch.sqrt((codeap_sqrsum / codeap_count) - (codeap_sum / codeap_count) ** 2),
    }
    print('saving...')
    torch.save(state_dict, output_path)


def cli_main():
    parser = ArgumentParser(description="Make stat file for TTS (audio_stat.pt)")
    parser.add_argument("--output", type=str, required=True)
    parser = AudioTextDataModule.add_argparse_args(parser)
    parser.set_defaults(vocoder="world")
    args = parser.parse_args()
    assert args.vocoder == "world" or args.vocoder == "world_mcep"
    data = AudioTextDataModule.from_argparse_args(args)
    data.setup("predict")
    calc_stat(data, args.output)


if __name__ == '__main__':
    cli_main()
