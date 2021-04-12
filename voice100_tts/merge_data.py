import numpy as np

def merge_data(src_list, dst, sample_rate):
    for tag in 'text', 'audio':
        with open(f'data/{dst}-{tag}-{sample_rate}.idx', 'wb') as outf:
            last_index = 0
            for src in src_list:
                indices = np.fromfile(f'data/{src}-{tag}-{sample_rate}.idx', dtype=np.int64)
                indices += last_index
                last_index = indices[-1]
                outf.write(bytes(memoryview(indices)))
        with open(f'data/{dst}-{tag}-{sample_rate}.bin', 'wb') as outf:
            for src in src_list:
                with open(f'data/{src}-{tag}-{sample_rate}.bin', 'rb') as f:
                    while True:
                        buf = f.read()
                        if not buf:
                            break
                        outf.write(buf)

merge_data(['cv_ja', 'kokoro_tiny'], 'cv_ja_kokoro_tiny', 16000)

def a():
    import os
    dataset = 'kokoro_tiny'
    tag = 'audio'
    sample_rate=16000
    os.rename(f'data/{dataset}-{tag}-{sample_rate}.idx', f'data/{dataset}-{tag}-{sample_rate}.idx.bak')
    indices = np.fromfile(f'data/{dataset}-{tag}-{sample_rate}.idx.bak', dtype=np.int32).astype(np.int64)
    with open(f'data/{dataset}-{tag}-{sample_rate}.idx', 'wb') as f:
        f.write(bytes(memoryview(indices)))