from tqdm import tqdm
import torch

from voice100.datasets import ASRDataModule
#data = ASRDataModule('kokoro_small', 0.1, 'ja', 10, 'cache', 5)
data = ASRDataModule('librispeech', 0.1, 'en', 10, 'cache', 128)
data.setup()

from voice100.audio import BatchSpectrogramAugumentation
augument = BatchSpectrogramAugumentation()

import random
for i in range(10):
    for batch in tqdm(data.train_dataloader()):
        #print(batch)
        with torch.no_grad():
            (audio, audio_len), (text, text_len) = batch
            audio1, audio_len1 = augument(audio, audio_len)
            if False:
                rate = 0.5 + random.random()
                i = (torch.arange(int(audio.shape[1] * rate), device=audio.device) / rate).int()
                audio1 = torch.index_select(audio, 1, i)
                audio_len1 = (audio_len * rate).int()
            assert not (torch.any(audio1.isnan()))
            assert (torch.max(audio_len1) > 0)
            assert (torch.max(audio_len1) == audio1.shape[1]), f'{audio_len1} {audio1.shape}'