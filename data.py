import numpy as np
import pickle
import torch
from torch.utils.data import Dataset


class DINDataset(Dataset):
    def __init__(self,
                 file_path=None,
                 max_genre_len=6,
                 max_hist_len=1001
                 ):
        super().__init__()
        assert file_path is not None
        with open(file_path, 'rb') as f:
            self.ds = pickle.load(f)
        self.max_genre_len = max_genre_len
        self.max_hist_len = max_hist_len

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        # trg_movie, trg_genre, user_id, sex, age, occupation, hist_movie, hist_genre, label
        tmp = self.ds[idx]
        trg_movie = tmp[0]
        trg_genre = tmp[1]
        user = (tmp[2], tmp[3], tmp[4], tmp[5])  # user_id, sex, age, occupation

        hist_movie = tmp[6]
        hist_genre = tmp[7]
        mask_id = [1] * len(hist_movie) + [0] * (self.max_hist_len - len(hist_movie))
        if len(hist_movie) < self.max_hist_len:
            hist_movie += [0] * (self.max_hist_len - len(hist_movie))
            hist_genre += [[0 for _ in range(self.max_genre_len)] for _ in range(self.max_hist_len - len(hist_genre))]

        label = tmp[8]

        assert len(hist_movie) == self.max_hist_len, len(hist_movie)
        assert len(hist_genre) == self.max_hist_len, len(hist_genre)
        assert len(mask_id) == self.max_hist_len, len(mask_id)

        seq = (
            user,
            trg_movie,
            trg_genre,
            hist_movie,
            hist_genre,
            mask_id,
            label
        )
        return tuple(map(lambda x: torch.tensor(x), seq))

