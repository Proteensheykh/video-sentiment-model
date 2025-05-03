import torch
import torch.nn as nn
from transformers import BertModel
from torchvision import models as vision_models


class TextEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

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
    def __init__(self) -> None:
        super().__init__()

        self.conv_layers = nn.Sequential(
            # lower order feature detection
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # higher order feature detection
            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        for param in self.conv_layers.parameters():
            param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, audio_features):
        audio_features = audio_features.squeeze(1)

        features = self.conv_layers(audio_features)
        
        # features output: [batch_size, 128, 1] - squeeze out the last dimension 
        return self.projection(features.squeeze(-1))
    

class MultiModalSentimentModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # Encoders
        self.text_encoder = TextEncoder()
        self.video_encoder = VideoEncoder()
        self.audio_encoder = AudioEncoder()

        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear((128 * 3), 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Classification heads(models)
        self.emotion_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 7) # sadness, anger, disgust,...
        )

        self.sentiment_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)
        )

def forward(self, text_inputs, video_frames, audio_features):
    text_features = self.text_encoder(
        text_inputs['input_ids'],
        text_inputs['attention_mask']
    )
    video_features = self.video_encoder(video_frames)
    audio_features = self.audio_encoder(audio_features)

    # concatenate multimodal features
    combined_features = torch.cat([
        text_features,
        video_features,
        audio_features
    ], dim=1)

    fused_features = self.fusion_layer(combined_features)

    emotion_output = self.emotion_classifier(fused_features)
    sentiment_output = self.sentiment_classifier(fused_features)

    return {
        'emotions': emotion_output,
        'sentiments': sentiment_output
    }
