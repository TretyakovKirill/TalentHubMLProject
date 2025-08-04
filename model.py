import math
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingCNN(nn.Module):
    def __init__(
        self,
        f1=16,
        expansion_factor=2,
        kernel_size1=63,
        kernel_size2=15,
        pooling_size1=8,
        pooling_size2=8,
        dropout_rate=0.3,
        n_channel=22,
        emb_size=40
    ):
        super().__init__()
        f2 = expansion_factor * f1

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, kernel_size1), padding='same', bias=False),
            nn.BatchNorm2d(f1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(f1, f2, (n_channel, 1), groups=f1, padding=0, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, pooling_size1)),
            nn.Dropout(dropout_rate)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(f2, f2, (1, kernel_size2), padding='same', bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, pooling_size2)),
            nn.Dropout(dropout_rate)
        )

        self.proj = nn.Linear(f2, emb_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.flatten(start_dim=2)
        x = x.transpose(1, 2)
        # x = self.proj(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000, dropout_rate=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ConformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, conv_kernel_size,
                 dropout_rate=0.1, activation="swish"):
        super().__init__()

        self.ffn1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.SiLU() if activation=="swish" else nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout_rate),
        )
        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout_rate,
                                         batch_first=True)
        self.norm_mha = nn.LayerNorm(d_model)

        self.norm_conv = nn.LayerNorm(d_model)
        self.conv_module = nn.Sequential(
            nn.Conv1d(d_model, 2*d_model, kernel_size=1),
            nn.GLU(dim=1),
            nn.Conv1d(d_model, d_model, kernel_size=conv_kernel_size,
                      padding=conv_kernel_size//2, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Dropout(dropout_rate),
        )
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.SiLU() if activation=="swish" else nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout_rate),
        )
        self.norm_final = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = residual + 0.5 * self.ffn1(x)

        residual = x
        x_mha, _ = self.mha(x, x, x, need_weights=False)
        x = residual + F.dropout(x_mha, p=self.mha.dropout, training=self.training)
        x = self.norm_mha(x)

        residual = x
        x_conv = self.norm_conv(x)
        x_conv = x_conv.transpose(1, 2)
        x_conv = self.conv_module(x_conv)
        x = residual + x_conv.transpose(1, 2)

        residual = x
        x = residual + 0.5 * self.ffn2(x)

        return self.norm_final(x)


class Model(nn.Module):
    def __init__(
        self,
        n_class=4,
        n_channel=22,
        cnn_f1=20,
        kernel_size1=63,
        kernel_size2=15,
        expansion_factor=2,
        pooling_size1=8,
        pooling_size2=8,
        dropout_rate=0.3,
        emb_size=40,
        heads=2,
        depth=3,
        conformer_ffn_dim=64,
        conformer_conv_kernel=31,
        conformer_dropout=0.1,
        flatten_strategy='avg',
        **kwargs
    ):
        super().__init__()

        self.embedding = EmbeddingCNN(
            f1=cnn_f1,
            kernel_size1=kernel_size1,
            kernel_size2=kernel_size2,
            expansion_factor=expansion_factor,
            pooling_size1=pooling_size1,
            pooling_size2=pooling_size2,
            dropout_rate=dropout_rate,
            n_channel=n_channel,
            emb_size=emb_size
        )

        self.pos_encoder = PositionalEncoding(d_model=emb_size,
                                              dropout=conformer_dropout)

        # self.transformers = TransformerEncoder(
        #     encoder_layer=TransformerEncoderLayer(
        #         d_model=emb_size,
        #         nhead=heads,
        #         dim_feedforward=conformer_ffn_dim,
        #         dropout=conformer_dropout,
        #     ),
        #     num_layers=depth
        # )

        self.conformers = nn.ModuleList([
            ConformerBlock(
                d_model=emb_size,
                nhead=heads,
                dim_feedforward=conformer_ffn_dim,
                conv_kernel_size=conformer_conv_kernel,
                dropout=conformer_dropout
            )
            for _ in range(depth)
        ])

        self.flatten_strategy = flatten_strategy
        if flatten_strategy == 'avg':
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif flatten_strategy == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        else:
            raise ValueError("Unsupported flatten_strategy: choose 'avg' or 'cls'")

        self.classifier = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_class)
        )

    def forward(self, x):
        x = self.embedding(x)

        if self.flatten_strategy == 'cls':
            b, _, _ = x.shape
            cls = self.cls_token.expand(b, -1, -1)
            x = torch.cat([cls, x], dim=1)

        x = self.pos_encoder(x)

        for block in self.conformers:
            x = block(x)

        if self.flatten_strategy == 'avg':
            x_pooled = x.transpose(1, 2)
            x_pooled = self.pool(x_pooled).squeeze(-1)
        else:
            x_pooled = x[:, 0, :]

        out = self.classifier(x_pooled)
        return x_pooled, out
