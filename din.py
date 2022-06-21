import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self,
                 dim):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(dim, 80),
            nn.Sigmoid(),
            nn.Linear(80, 40),
            nn.Sigmoid(),
            nn.Linear(40, 1)
        )
        self.act = nn.Softmax(dim=-1)

    def forward(self,
                queries,
                keys,
                mask=None,
                ):
        """
        queries: [B, H]
        keys: [B, T, H]
        """
        hidden_dim = queries.shape[-1]
        queries = torch.tile(queries, [1, keys.shape[1]])
        queries = queries.reshape(-1, keys.shape[1], hidden_dim)
        din_all = torch.cat([queries, keys, queries - keys, queries * keys], -1)  # (B, T, 4H)
        outputs = self.layer(din_all)  # (B, T, 1)
        outputs = outputs.transpose(2, 1)  # (B, 1, T)
        outputs = outputs.squeeze()
        if mask is not None:
            outputs = outputs.masked_fill(mask == 0, (-2 ** 32 + 1))
        outputs = outputs.unsqueeze(dim=1)
        outputs = outputs / (keys.shape[1] ** 0.5)
        scores = self.act(outputs)
        # print(f"scores:{scores}")
        outputs = torch.matmul(scores, keys)  # (B, 1, H)

        return outputs


class Dice(nn.Module):
    def __init__(self,
                 emb_size,
                 dim=2,
                 epsilon=1e-8,
                 device='cpu'
                 ):
        super().__init__()
        assert dim == 2 or dim == 3
        self.bn = nn.BatchNorm1d(emb_size, eps=epsilon)
        self.sigmoid = nn.Sigmoid()
        self.dim = dim
        if self.dim == 2:
            self.alpha = nn.Parameter(torch.zeros((emb_size,)).to(device))
        else:
            self.alpha = nn.Parameter(torch.zeros((emb_size, 1)).to(device))

    def forward(self, x):
        assert x.dim() == self.dim
        if self.dim == 2:
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
        else:
            x = torch.transpose(x, 1, 2)
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
            out = out.transpose(1, 2)
        return out


class DIN(nn.Module):
    def __init__(self,
                 num_users=6040,
                 num_movies=3706,
                 num_gender=2,
                 num_occupation=21,
                 num_age=7,
                 num_genres=18+1,  # padding
                 hidden_dim=128,
                 max_genres=6,
                 device='cpu'
                 ):
        super().__init__()
        self.us_emb = nn.Embedding(num_users, hidden_dim // 4)
        self.mo_emb = nn.Embedding(num_movies, hidden_dim // 2)
        self.ca_emb = nn.Embedding(num_genres, hidden_dim // 2)
        self.ge_emb = nn.Embedding(num_gender, hidden_dim // 4)
        self.oc_emb = nn.Embedding(num_occupation, hidden_dim // 4)
        self.ag_emb = nn.Embedding(num_age, hidden_dim // 4)

        self.cate_pool = nn.MaxPool2d((1, max_genres))

        self.attention = Attention(dim=hidden_dim * 4)
        self.b_norm = nn.BatchNorm1d(hidden_dim)
        self.att_linear = nn.Linear(hidden_dim, hidden_dim)

        self.b1 = nn.BatchNorm1d(hidden_dim * 3)
        self.d_layer = nn.Sequential(
            nn.Linear(hidden_dim * 3, 80),
            Dice(80, device=device),
            nn.Linear(80, 40),
            Dice(40, device=device),
            nn.Linear(40, 2)
        )

    def forward(self,
                user,  # (B, 4) user_id, gender, age, occupation
                trg_movie,  # (B,)
                trg_cate,  # (B, max_genre)
                hist_movie,  # (B, max_hist)
                hist_genre,  # (B, max_hist, max_genre)
                mask_id,  # (B, max_hist)
                label=None,
                return_loss=False
                ):
        uid_emb = self.us_emb(user[:, 0].reshape(-1, 1)).squeeze()
        ge_emb = self.ge_emb(user[:, 1].reshape(-1, 1)).squeeze()
        oc_emb = self.oc_emb(user[:, 3].reshape(-1, 1)).squeeze()
        ag_emb = self.ag_emb(user[:, 2].reshape(-1, 1)).squeeze()
        user_emb = torch.cat((uid_emb, ge_emb, oc_emb, ag_emb), 1)

        trg_mo_emb = self.mo_emb(trg_movie).squeeze()
        trg_ca_emb = self.ca_emb(trg_cate)  # (B, C, H)
        trg_ca_emb = self.cate_pool(trg_ca_emb.transpose(2, 1)).squeeze()
        trg_emb = torch.cat([trg_mo_emb, trg_ca_emb], 1)  # (B, H)

        hist_mo_emb = self.mo_emb(hist_movie)  # (B, T, H)
        hist_ca_emb = self.ca_emb(hist_genre)  # (B, T, C, H)
        hist_ca_emb = self.cate_pool(hist_ca_emb.transpose(3, 2)).squeeze()
        hist_emb = torch.cat([hist_mo_emb, hist_ca_emb], 2)  # (B, T, H)

        hist_i = self.attention(trg_emb, hist_emb, mask_id)
        hist_i = self.b_norm(hist_i.squeeze())
        hist_i = self.att_linear(hist_i)  # (B, H)

        din_i = torch.cat([user_emb, hist_i, trg_emb], -1)
        din_i = self.b1(din_i)
        logits = self.d_layer(din_i).squeeze()

        if not return_loss:
            return logits

        if label is not None:
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, label)
            return loss
