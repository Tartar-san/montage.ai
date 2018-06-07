from pytube import YouTube
import os

with open("videos_to_scrap.txt") as f:
    for line in f.readlines():
        url = line.rstrip()
        if YouTube(url).streams.first().default_filename not in os.listdir("scrapped_videos"):
            YouTube(url).streams.first().download("scrapped_videos")
            print("Downloaded: " + url)
        else:
            print("It is already downloaded: " + url)


