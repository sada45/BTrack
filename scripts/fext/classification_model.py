import torch.nn as nn
import torch.nn.functional as F
import config

class FeatureExtractor(nn.Module):
    def __init__(self, class_num, in_chan=6, feat_len=config.feat_len):
        super(FeatureExtractor, self).__init__()

        cnn_output_size = 16
        lstm_hidden_size = 16
        disc_fc_size = 64

        self.conv = nn.Sequential(
            # CNN 1
            nn.Conv1d(in_channels=in_chan, out_channels=64, kernel_size=5, padding=2, dilation=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            # CNN 2
            nn.Conv1d(64, 64, kernel_size=7, padding=3, dilation=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            # CNN 3
            nn.Conv1d(64, 64, kernel_size=5, padding=4, dilation=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            # CNN 4
            nn.Conv1d(64, 64, kernel_size=5, padding=8, dilation=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            # CNN 5
            nn.Conv1d(64, 64, kernel_size=5, padding=16, dilation=8),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            # CNN 6
            nn.Conv1d(64, 64, kernel_size=5, padding=32, dilation=16),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            # CNN 7
            nn.Conv1d(64, cnn_output_size, kernel_size=1, dilation=1),
            nn.BatchNorm1d(cnn_output_size),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(input_size=cnn_output_size, hidden_size=lstm_hidden_size, batch_first=True, bidirectional=True)

        self.linear = nn.Sequential(
            nn.Linear(2 * lstm_hidden_size * feat_len, disc_fc_size),
            # nn.Linear(cnn_output_size * feat_len, disc_fc_size),
            nn.BatchNorm1d(disc_fc_size),
            nn.ReLU(),

            nn.Linear(disc_fc_size, disc_fc_size),
            nn.BatchNorm1d(disc_fc_size),
            nn.ReLU(),

            nn.Linear(disc_fc_size, disc_fc_size),
            nn.BatchNorm1d(disc_fc_size),
            nn.ReLU(),

            nn.Linear(disc_fc_size, disc_fc_size),
            nn.BatchNorm1d(disc_fc_size),
            nn.ReLU(),
        )

        self.final_linear = nn.Linear(disc_fc_size, class_num)
    
    def forward(self, x):
        out = x.contiguous()  # [B, 1, L], L=400
        out = self.conv(out)  # [B, cnn_output_size, L]
        out = out.transpose(1, 2).contiguous()  # [B, L, cnn_output_size]
        out, _ = self.lstm(out)  # [B, L, 2 * lstm_hidden_size]
        out = F.relu(out).contiguous()
        out = out.view(out.size(0), -1).contiguous()  # [B, L * 2 * lstm_hidden_size]
        out = self.linear(out)
        final_out = self.final_linear(out)
        return final_out, out

