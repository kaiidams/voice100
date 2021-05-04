# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import numpy as np

class IndexDataFileWriter:
    def __init__(self, file):
        self.file = file
        self.current = 0
        self.indices = []
        self.data = []

    def __enter__(self):
        self.idx_f = open(f'{self.file}.idx', 'wb')
        self.bin_f = open(f'{self.file}.bin', 'wb')
        return self

    def write(self, data):
        self.current += len(data)
        self.idx_f.write(bytes(memoryview(np.array(self.current, dtype=np.int64))))
        self.bin_f.write(data)

    def __exit__(self, exc_type, exc_value, traceback):
        self.idx_f.close()
        self.bin_f.close()

def open_index_data_for_write(prefix):
    return IndexDataFileWriter(prefix)

class IndexDataFileReaderV1:
    def __init__(self, file):
        self.indices = np.fromfile(file + '.idx', dtype=np.int64)
        self.data = np.fromfile(file + '.bin', dtype=np.uint8)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        start = self.indices[index - 1] if index > 0 else 0
        end = self.indices[index]
        return self.data[start:end]

class IndexDataFileReader:
    def __init__(self, file):
        import mmap
        self.indices = np.fromfile(file + '.idx', dtype=np.int64)
        self.file_obj = open(file + '.bin', mode="rb")
        self.data = mmap.mmap(self.file_obj.fileno(), length=0, access=mmap.ACCESS_READ)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        start = self.indices[index - 1] if index > 0 else 0
        end = self.indices[index]
        return self.data[start:end]

    def close(self):
        self.data.close()
        self.file_obj.close()

class IndexDataDataset:
    def __init__(self, readers_or_files, shapes, dtypes, dups=None):
        self.readers = [
            IndexDataDataset._getreader(reader_or_file)
            for reader_or_file in readers_or_files
        ]
        self.shapes = shapes
        self.dtypes = dtypes
        self.indices = None
        if dups:
            self.dups = dups
        else:
            self.dups = [1] * len(readers_or_files)

    def split(self, weights):
        slots = []
        for i, weight in enumerate(weights):
            slots.extend([i] * weight)
        res = []
        for i in range(len(weights)):
            ds = IndexDataDataset(self.readers, self.shapes, self.dtypes, dups=self.dups)
            ds.indices = list(filter(lambda j: slots[j % len(slots)] == i, range(len(self))))
            res.append(ds)
        return res

    @staticmethod
    def _getreader(reader_or_file):
        if isinstance(reader_or_file, str):
            return IndexDataFileReader(reader_or_file)
        return reader_or_file

    def __len__(self):
        return len(self.indices) if self.indices else len(self.readers[0])

    def __getitem__(self, index):
        if self.indices:
            index = self.indices[index]
        return [
            np.frombuffer(reader[index // dup], dtype=dtype).reshape(shape)
            for reader, shape, dtype, dup in zip(self.readers, self.shapes, self.dtypes, self.dups)
        ]

    def close(self):
        for reader in self.readers:
            reader.close()