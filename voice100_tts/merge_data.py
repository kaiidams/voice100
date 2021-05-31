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

def fix_index():
    import os
    import sys
    file = sys.argv[1] + '.idx'
    assert not os.path.exists(file + '.bak')
    os.rename(file, file + '.bak')
    indices = np.fromfile(file + '.bak', dtype=np.int32).astype(np.int64)
    assert indices[0] != 0
    assert indices[1] != 0
    assert indices[2] != 0
    assert indices[3] != 0
    with open(file, 'wb') as f:
        f.write(bytes(memoryview(indices)))