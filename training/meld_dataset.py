from torch.utils.data import Dataset, DataLoader
import torch.utils.data.dataloader
from transformers import AutoTokenizer
import cv2
import numpy as np
import os
import pandas as pd
import subprocess
import torch
import torchaudio

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class MELDDataset(Dataset):

    def __init__(self, csv_path, video_dir) -> None:
        self.data = pd.read_csv(csv_path)
        # print(f"CSV loaded..., {len(self.data)}")
        self.video_dir = video_dir
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.emotion_map = {
            'anger': 0, 
            'disgust': 1, 
            'fear': 2, 
            'joy': 3, 
            'neutral': 4, 
            'sadness': 5, 
            'surprise': 6
            }
        self.sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}

    def _load_video_frames(self, video_path: str):
        # load capture
        capture = cv2.VideoCapture(video_path)
        frames = []

        try:
            if not capture.isOpened():
                raise ValueError(f"Video not found: {video_path}")
    
            # try and read 1st frame to validate video/capture
            ret, frame = capture.read()
            if not ret or frame == None:
                raise ValueError(f"Video not found: {video_path}")

            # reset index after test read(to validate)
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # itrate and read the next 30 frames(30frames ~ 1sec)
            while len(frames) and capture.isOpened():

                ret, frame = capture.read()
                if not ret:
                    break
                # resize frame to 224x224
                frame = cv2.resize(frame, (224,224))

                # normalize RGB (frame/225)
                frame = frame / 225.0
                frames.append(frame)
            
            if len(frames) <= 0:
                raise ValueError(f"Video Error: No frames found in capture")

            # pad & truncate frames
            if len(frames) < 30: # standard frame rate - 30fps
                frames += [np.zeros_like(frames[0]) * (30 - len(frames))]
            else:
                frames = frames[:30]

            # Before permute: [frame, height, width, channels]
            # After permute: [frame, channels, heigth, width] - RESNET3D specification
            return torch.FloatTensor(np.array(frames)).permute(0,3,1,2)

        except Exception as e:
            raise ValueError(f"Video Error: {str(e)}")
        
        finally:
            capture.release()
    
    def _extract_audio_features(self, video_path: str):
        audio_path = video_path.removesuffix('.mp4', '.wav')

        try:
            # extract audio file(.wav) from video mp4
            subprocess.run([
                'ffmpeg', # must be installed on local machine(device)
                '-i', video_path,
                '-vn', # no video(audio only)
                '-acodec', 'pcm_s16le',
                '-ar', '16000', # sample/audio rate
                '-ac', '1', # audio channel - [monochannel, duochannel(i.e headphones),...]
                audio_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # load audio file(.wav)
            waveform, sample_rate = torchaudio.load(audio_path)

            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)

            mel_spectogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_mels=64,
                n_fft=1024,
                hop_length=512
            )

            mel_spec = mel_spectogram(waveform)

            # Normalize mel-spectogram
            mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()

            if mel_spec.size(2) < 300: # time_steps
                padding = 300 - mel_spec.size(2)
                mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
            else:
                mel_spec = mel_spec[:, :, :300]

            return mel_spec

        except subprocess.CalledProcessError as e:
            raise ValueError(f"Audio estraction error: {str(e)}")
        except Exception as e:
            raise ValueError(f"Audio error: {str(e)}")
        finally:
            if os.path.exists[audio_path]:
                os.remove(audio_path)

    def __len__(self) -> int:
        return len(self.data)
    
    def __getItem__(self, idx: int):

        if isinstance(idx, torch.Tensor): # handle tensor argument
            idx = idx.item()

        row = self.data.iloc[idx]
        try:
            # Get video frames
            filename = f"""dia{row['Dialogue_ID']}utt{row['Utterance_ID']}.mp4"""
            path = os.path.join(self.video_dir, filename)
            path_exists = os.path.exists(path)

            if not path_exists:
                raise FileNotFoundError(f"No file was found: {path}")
            
            video_frames = self._load_video_frames(path)
            
            # Get text input
            text_inputs = self.tokenizer(
                row['Utterance'],
                padding='max_length',
                truncation=True,
                max_length=128,
                return_tensors='pt'
                )
            
            # Get Audio
            audio_features = self._extract_audio_features(path)

            # Map sentiment and emottion labels
            sentiment_label = self.sentiment_map[row['Sentiment'].lower()]
            emotion_label = self.emotion_map[row['Emotion'].lower()]

            return {
                'text_inputs': {
                    'input_ids': text_inputs['input_ids'].squeeze(),
                    'attention_mask': text_inputs['attention_mask'].squeeze()
                },
                'video_frames': video_frames,
                'audio_features': audio_features,
                'sentiment_label': torch.tensor(sentiment_label),
                'emotion_label': torch.tensor(emotion_label)
            }
        
        except Exception as e:
            print(f"Error processing - {path}: {str(e)}")
            return None
        

def collate_fn(batch):
    # filter out 'None' values from batch
    # because dataset[i](a.k.a __getItem__) returns None for failed item processing
    batch = list(filter(lambda x: x is not None, batch)) 
    return torch.utils.data.dataloader.default_collate(batch)

    
def prepare_dataloaders(train_csv, train_video_dir,
                        dev_csv, dev_video_dir,
                        test_csv, test_video_dir, batch_size=32):
    
    train_dataset = MELDDataset(train_csv, train_video_dir)
    dev_dataset = MELDDataset(dev_csv, dev_video_dir)
    test_dataset = MELDDataset(test_csv, test_video_dir)

    train_loader = DataLoader(train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset,
                            batch_size=batch_size,
                            collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset,
                            batch_size=batch_size,
                            collate_fn=collate_fn)
    
    return train_loader, dev_loader, test_loader


if __name__ == "__main__":
    train_loader, dev_loader, test_loader = prepare_dataloaders(
        '../path/to/train/csv', '../path/to/train/video_splits',
        '../path/to/dev/csv', '../path/to/dev/video_splits',
        '../path/to/test/csv', '../path/to/test/video_splits',
    )

    # test sample data
    for batch in train_loader:
        print(batch['text_inputs'])
        print(batch['video_frames'].shape)
        print(batch['audio_features'].shape)
        print(batch['emotion_label'])
        print(batch['sentiment_label'])
        break