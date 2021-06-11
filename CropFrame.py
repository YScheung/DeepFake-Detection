import cv2
import pytube
import os
import csv
import pytube.exceptions

linklist = []
with open('urls_fake.csv') as f:
    data = csv.reader(f)
    for row in data:
        print(row[0])
        linklist.append(row[0])

fcount=0
count=367800
for url in linklist:
    try:
        youtube = pytube.YouTube(url)
        video = youtube.streams.get_highest_resolution()
        video.download(filename=str(fcount))
        name = str(fcount) + '.mp4'
        vidcap = cv2.VideoCapture(name)
        success,image = vidcap.read()
        while success:
            success,image = vidcap.read()
            if (count % 200 == 0):
              cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
            count += 1
    except pytube.exceptions.VideoUnavailable:
        print("Failed")
    fcount += 1
