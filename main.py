from ChunkExtractor import ChunkExtractor
from VideoFeatureExtractor import VideoFeatureExtractor
from AudioAnalyzer import AudioAnalyzer
from ChunksClustering import ChunksClustering
from moviepy.editor import *
from config import *
import argparse
import random
import os
import numpy as np

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--music", required=True, help="Path to the audio, we want to generate video for")
    ap.add_argument("-v", "--video_folder", required=True, help="Path to the folder which contains input videos")
    ap.add_argument("-o", "--output_path", default="result.mp4", help="Path to save generated video")
    args = vars(ap.parse_args())
    print("Cleaning before making chunks...")
    # shutil.rmtree(CHUNKS_PATH)
    # os.mkdir(CHUNKS_PATH)
    print("Analyzing audio...")
    audio_analyzer = AudioAnalyzer(args["music"])
    audio_analyzer.run()
    print("Extracting chunks from the videos...")
    min_chunk, max_chunk = audio_analyzer.get_min_max_intervals()


    videos = list(map(lambda x: os.path.join(args["video_folder"], x), os.listdir(args["video_folder"])))

    # list in format [path_to_the_chunk, duration]
    chunks = []

    for video in videos:
        chunker = ChunkExtractor(min_chunk, max_chunk)
        chunks.extend(chunker.extract_chunks(video))
    # print(chunks)
    # lll = input("wait")
    print("Extracting visual features from the chunks...")
    v = VideoFeatureExtractor()
    chunks_and_visual_features = [(x, v.extract_features_from_video(x[0])) for x in chunks]
    # print(chunks_and_visual_features)
    # input("wait")

    timestamps = audio_analyzer.get_timestamps()
    sec_timestamps = [i / 1000 for i in timestamps]
    chunk_audio_features = audio_analyzer.get_chunks_features()

    print("Generating video...")
    c = ChunksClustering()

    scenes_sequence = []

    last_time = 0
    end_time = audio_analyzer.get_duration()
    for time, audio_features in zip(sec_timestamps, chunk_audio_features):
        possible_chunks = list(filter(lambda x: x[0][1] >= (time-last_time)*1000, chunks_and_visual_features))
        # print(possible_chunks)
        # aaa = input("possible chunks")

        if len(possible_chunks) == 0:
            break

        chunks_audio_video_features = [np.concatenate((x[1], audio_features)) for x in possible_chunks]
        clusters = [c.predict_cluster(ftrs) for ftrs in chunks_audio_video_features]
        # print(clusters)
        # aaa = input("clusters")
        maxdist = max([i[1] for i in clusters])
        dist_clusters = [i[1] for i in clusters]
        # print(dist_clusters)
        # aaa = input("dist_clusters")
        chunk_idx = np.argmin(dist_clusters)
        # print(possible_chunks[0])
        chunk = possible_chunks[chunk_idx]
        # print(chunk)
        scene = VideoFileClip(chunk[0][0], audio=False).set_start(last_time).set_end(time)
        scenes_sequence.append(scene)
        chunks_and_visual_features.remove(chunk)
        last_time = time

    possible_chunks = list(filter(lambda x: x[1] >= (end_time-last_time)*1000, chunks))

    if len(possible_chunks) > 0:
        chunk = random.choice(possible_chunks)
        scene = VideoFileClip(chunk[0], audio=False).set_start(last_time).set_end(end_time)
        scenes_sequence.append(scene)

    audio_track = AudioFileClip(args["music"])
    music_video = CompositeVideoClip(scenes_sequence)
    music_video = music_video.set_audio(audio_track)
    music_video.write_videofile(args["output_path"])


