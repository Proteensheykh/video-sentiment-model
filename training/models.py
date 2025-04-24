import torch.nn as nn
from transformers import BertModel
from torchvision import models as vision_models

class TextEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncase')

        # exclude pre-trained model params from training.
        # Training is much faster with fewer trainable nodes
        for param in self.bert.parameters():
            param.requires_grad = False  # marks parameters as unavailable for training

        self.projection = nn.Linear(768, 128)

    def forward(self, input_ids, attention_mask):
        # Extract Bert Embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token representation
        pooler_output = outputs.pooler_output

        return self.projection(pooler_output)

class VideoEncoder(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.backbone = vision_models.video.r3d_18(pretrained=True)

        for param in self.backbone.parameters():
            param.requires_grad = False

        num_fts = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_fts, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, video_frames):
        video_frames = video_frames.transpose(1, 2)
        return self.backbone(video_frames)

class AudioEncoder(nn.Module):
    pass