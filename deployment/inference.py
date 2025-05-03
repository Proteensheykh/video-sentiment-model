import torch
import os, subprocess
import cv2, torchaudio
import numpy as np
import whisper

from transformers import AutoTokenizer
from models import MultiModalSentimentModel


EMOTION_MAP = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'joy', 4: 'neutral', 5: 'sadness', 6: 'surprise'}
SENTIMENT_MAP = {0: 'negative', 1: 'neutral', 2: 'positive'}

class VideoProcessor:
    def process_videos(self, video_path: str):
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


class AudioProcessor:
    def extract_features(self, video_path: str, max_length=300):
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

            if mel_spec.size(2) < max_length: # time_steps
                padding = max_length - mel_spec.size(2)
                mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
            else:
                mel_spec = mel_spec[:, :, :max_length]

            return mel_spec

        except subprocess.CalledProcessError as e:
            raise ValueError(f"Audio estraction error: {str(e)}")
        except Exception as e:
            raise ValueError(f"Audio error: {str(e)}")
        finally:
            if os.path.exists[audio_path]:
                os.remove(audio_path)

class VideoUtteranceProcessor:

    def __init__(self) -> None:
        self.video_processor = VideoProcessor()
        self.audio_processor = AudioProcessor()

    def extract_segment(self, video_path, start_time, end_time, temp_dir="/temp"):
        os.mkdir(temp_dir, exist_ok=True)
        segment_path = os.path.join(temp_dir, f"segment_{start_time}_{end_time}.mp4")

        subprocess.run([
            "ffmpeg",
            "-ss", start_time,
            "to", end_time,
            "-c:v", "libx264",
            "-c:a", "aac",
            "y",
            segment_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if not os.path.exists(segment_path) or os.path.getsize(segment_path) == 0:
            raise ValueError("Segment extraction failed:", segment_path)
        
        return segment_path

def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModalSentimentModel().to(device)

    model_path = os.path.join(model_dir, "model.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, "model", "model.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model file not found in directory:", model_dir)
        
    print("Loading model from path: ", model_path)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    return {
        'model': model,
        'tokenizer': AutoTokenizer.from_pretrained('bert-base-uncased'),
        'transcriber': whisper.load_model(
            "base",
            device="cpu" if device.type == "cpu" else device, 
        ),
        'device': device
    }

def predict_fn(input_data, model_dict):
    model, tokenizer, transcriber, device = model_dict.values()
    video_path = input_data['video_path']

    result = transcriber.transcribe(video_path, word_timestamps=True)

    utterance_processor = VideoUtteranceProcessor()
    predictions = []

    for segment in result["segments"]:
        try:
            segment_path = utterance_processor.extract_segment(
                video_path=video_path,
                start_time=segment['start'],
                end_time=segment['end']
            )

            video_frames = utterance_processor.video_processor.process_videos(segment_path)
            audio_features = utterance_processor.audio_processor.extract_features(segment_path)
            text_inputs = tokenizer(
                segment['text'],
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )

            # move to device
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            video_frames = video_frames.unsqueeze(0).to(device)
            audio_features = audio_features.unsqueeze(0).to(device)

            # get predictions
            with torch.inference_mode():
                outputs = model(text_inputs, video_frames, audio_features)
                emotion_probs = torch.softmax(outputs['emotions'], dim=1)[0]
                sentiment_probs = torch.softmax(outputs['sentiments'], dim=1)[0]

                emotion_values, emotion_indices = torch.topk(emotion_probs, 3)
                sentiment_values, sentiment_indices = torch.topk(sentiment_probs, 3)
                
                predictions.append({
                    "start_time": segment['start'],
                    "end_time": segment['end'],
                    "emotions": [
                        {
                            "label": EMOTION_MAP[idx.item()], 
                            "confidence": conf.item()
                        } for idx, conf in zip(emotion_indices, emotion_values)
                    ],
                    "sentiments": [
                        {
                            "label": SENTIMENT_MAP[idx.item()], 
                            "confidence": conf.item()
                        } for idx, conf in zip(sentiment_indices, sentiment_values)
                    ]
                })
        except Exception as e:
            print("Segment failed inference...", str(e))

        finally:
            # cleanup
            if os.path.exists(segment_path):
                os.remove(segment_path)
    return {"utterances": predictions}    

def process_local_video(video_path, model_dir="model"):
    model_dict = model_fn(model_dir)
    input_data = {"video_path": video_path}

    predictions = predict_fn(input_data, model_dict)

    for utterance in predictions["utterance"]:
        print("\nUtterance:")
        print(f"Start: {utterance['start_time']}s, End: {utterance['end_time']}s")
        print(f"Utterance: {utterance['text']}")

        print("\nTop Emotions")
        for emotion in utterance['emotions']:
            print(f"{emotion['label']}: {emotion['confidence']:.2f}")

        print("\n TopSentiments")
        for sentiment in utterance['sentiments']:
            print(f"{sentiment['label']}: {sentiment['confidence']:.2f}")
        print('-'*50)

if __name__ == "__main__":
    process_local_video("./joy.mp4")
    