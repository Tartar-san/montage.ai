import libs.librosa as librosa
import ffmpy
import os
import numpy as np
from AudioAnalyzer import AudioAnalyzer
from VideoFeatureExtractor import VideoFeatureExtractor


a = AudioAnalyzer("")
v = VideoFeatureExtractor()

chunks_path = "chunks_scrapped_videos"
audio_chunks_path = "chunks_scrapped_videos_audio"

features = None

videos = map(lambda x: os.path.join("scrapped_videos", x),  os.listdir("scrapped_videos"))

print("CHUNKING...")
for video_path in videos:
    # first split video into segments
    print(video_path)
    chunk_video_path = os.path.join(chunks_path, os.path.basename(video_path))
    chunk_video_path = chunk_video_path.split(".")[0]+"%03d." +  chunk_video_path.split(".")[1]
    ff = ffmpy.FFmpeg(
        inputs={video_path: None},
        outputs={chunk_video_path: ['-c', 'copy', '-map', '0', '-segment_time', '8', '-f', 'segment', "-y"]}
    )
    ff.run()

chunks = map(lambda x: os.path.join(chunks_path, x), os.listdir(chunks_path))
print("EXTRACTING...")
processed = 0

for video_path in list(chunks):

    print(video_path)
    visual_features = v.extract_features_from_video(video_path)
    audio_path = os.path.join(audio_chunks_path, os.path.basename(video_path.split(".")[0]+".wav"))
    ff = ffmpy.FFmpeg(
        inputs={video_path: None},
        outputs={audio_path:["-acodec", "pcm_s16le", "-ac", "1", "-y"]}
    )
    ff.run()
    x, sr = librosa.load(audio_path)
    audio_features = a.extract_features(x, sr)

    if features is None:
        features = np.concatenate((visual_features, audio_features))
    else:
        features = np.vstack((features, np.concatenate((visual_features, audio_features))))
    processed +=1


print(features)
print("PROCESSED: ", processed)
np.save("music_videos_features.npy", features)

