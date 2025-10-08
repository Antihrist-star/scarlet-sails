import torch
import torch.nn as nn

class TimeEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.hour_emb = nn.Embedding(24, 8)
        self.dow_emb = nn.Embedding(7, 4)
        self.dom_emb = nn.Embedding(31, 6)
        
    def forward(self, hour, dow, dom):
        h = self.hour_emb(hour)
        d = self.dow_emb(dow)
        m = self.dom_emb(dom)
        return torch.cat([h, d, m], dim=1)  # 18 dims total

def add_time_features(df):
    """Добавить временные features в датафрейм"""
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    return df