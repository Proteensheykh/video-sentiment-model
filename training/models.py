import torch
import torch.nn as nn
from transformers import BertModel
from torchvision import models as vision_models
from sklearn.metrics import precision_score, accuracy_score

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
            nn.AdaptiveAvgPool1d
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
    def __init__(self):
        super.__init__()

        # Encoders
        self.text_encoder = TextEncoder()
        self.video_encoder = VideoEncoder()
        self.audio_encoder = AudioEncoder()

        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear((128 * 3), 256),
            nn.BatchNorm1d(256),
            nn.ReLu(),
            nn.Dropout(0.2)
        )

        # Classification heads(models)
        self.emotion_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.Relu(),
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


class MultiModalTrainer:

    def __init__(self, model, train_loader, val_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader

        #Log dataset sizes
        train_size = len(self.train_loader.dataset)
        val_size = len(self.val_loader.dataset)

        print("\nDataset sizes")
        print(f"Training dataset: {train_size}")
        print(f"Validation dataset: {val_size}")
        print(f"Batches per epoch: {len(self.train_loader)}")

        self.optimizer = torch.optim.Adam([
            {'params': model.text_encoder.parameters(), 'lr': 8e-6},
            {'params': model.audio_encoder.parameters(), 'lr': 8e-5},
            {'params': model.video_encoder.parameters(), 'lr': 8e-5},
            {'params': model.fusion_layer.parameters(), 'lr': 5e-4},
            {'params': model.emotion_classifier.parameters(), 'lr': 5e-4},
            {'params': model.sentiment_classifier.parameters(), 'lr': 5e-4},
        ], weight_decay=1e-5)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.1,
            patience=2
        )

        self.emotion_criterion = nn.CrossEntropyLoss(
            label_smoothing=0.05
        )

        self.sentiment_criterion = nn.CrossEntropyLoss(
            label_smoothing=0.05
        )

    def train_epoch(self):
        self.model.train()

        # accumulated running loss
        running_loss = {'total': 0, 'emotion': 0, 'sentiment': 0}

        for batch in self.train_loader:
            device = next(self.model.parameters()).to(device)
            text_inputs = {
                'input_ids': batch['text_inputs']['input_ids'].to(device),
                'attention_mask': batch['text_inputs']['attention_mask'].to(device)
            }
            video_frames = batch['video_frames'].to(device)
            audio_features = batch['audio_features'].to(device)
            emotion_labels = batch['emotion_labels'].to(device)
            sentiment_labels = batch['sentiment_labels'].to(device)

            #zero grad
            self.optimizer.zero_grad()

            # forward pass - calculate loss
            outputs = self.model(text_inputs, video_frames, audio_features)

            emotion_loss = self.emotion_criterion(outputs['emotions'], emotion_labels)
            sentiment_loss = self.sentiment_criterion(outputs['sentiments'], sentiment_labels)
            total_loss = emotion_loss + sentiment_loss

            # Backward pass - Calculate gradient & Optimize
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0)
            
            # Optimize parameters
            self.optimizer.step()

            # Track Losses
            running_loss['total'] += total_loss.item()
            running_loss['emotion'] += emotion_loss.item()
            running_loss['sentiment'] += sentiment_loss.item()

        # return the average loss-values for the entire training dataset
        return {k: v/len(self.train_loader) for k, v in running_loss.items()}
    
    def evaluate(self, data_loader, mode='val'):
        # turn off training mode
        self.model.eval()

        losses = {'total': 0, 'emotion': 0, 'sentiment': 0}
        all_emotion_predictions = []
        all_emotion_labels = []
        all_sentiment_predictions = []
        all_sentiment_labels = []

        with torch.inference_mode(): # turns off gradient calculation, makes inference computation faster
            for batch in data_loader:
                device = next(self.model.parameters()).to(device)
                text_inputs = {
                    'input_ids': batch['text_inputs']['input_ids'].to(device),
                    'attention_mask': batch['text_inputs']['attention_mask'].to(device)
                }
                video_frames = batch['video_frames'].to(device)
                audio_features = batch['audio_features'].to(device)
                emotion_labels = batch['emotion_labels'].to(device)
                sentiment_labels = batch['sentiment_labels'].to(device)

                # forward pass - calculate loss
                outputs = self.model(text_inputs, video_frames, audio_features)

                emotion_loss = self.emotion_criterion(outputs['emotions'], emotion_labels)
                sentiment_loss = self.sentiment_criterion(outputs['sentiments'], sentiment_labels)
                total_loss = emotion_loss + sentiment_loss

                
                all_emotion_predictions.extend( # aggregate the max value for each sample into a list
                    outputs['emotions'].argmax(dim=1).cpu().numpy())
                all_emotion_labels.extend(
                    emotion_labels.cpu().numpy())
                all_sentiment_predictions.extend( # aggregate the max value for each sample into a list
                    outputs['sentiments'].argmax(dim=1).cpu().numpy())
                all_sentiment_labels.extend(
                    sentiment_labels.cpu().numpy())
                
                # Track losses
                losses['total'] += total_loss.item()
                losses['emotion'] += emotion_loss.item()
                losses['sentiment'] += sentiment_loss.item()

                avg_loss = {k: v/len(data_loader) for k, v in losses.items()}

                # Compute the precision and accuracy
                emotion_precision = precision_score(
                    all_emotion_labels, all_emotion_predictions, 'average=weighted')
                # accuracy = right-predictions/total-predictions
                emotion_accuracy = accuracy_score(
                    all_emotion_labels, all_emotion_predictions)
                sentiment_precision = precision_score(
                    all_sentiment_labels, all_sentiment_predictions, 'average=weighted')
                sentiment_accuracy = accuracy_score(
                    all_sentiment_labels, all_sentiment_predictions)
                
                # Adjust the models learning-rate in neccessary, 
                # based on its performance on the validation dataset
                if mode == "val":
                    self.scheduler.step(avg_loss['total'])

                return avg_loss, {
                    'emtion_precision': emotion_precision,
                    'emotion_accuracy': emotion_accuracy,
                    'sentiment_precision': sentiment_precision,
                    'sentiment_accuracy': sentiment_accuracy
                }
