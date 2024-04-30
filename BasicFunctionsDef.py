"""
This module defines the basic functions of locating the water height.
"""
import statistics
import cv2
import os
import numpy as np
import pandas as pd
import tqdm
import time


# Define a function to extract frames from a video
def extract_frames(video_path):
    start = time.time()

    output_folder = 'frames'
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0

    with tqdm.tqdm(total=total_frames, ncols=90, desc='Extracting...', unit='images') as pbar:
        while True:
            success, frame = video.read()
            if not success:
                break
            cv2.imwrite(f"{output_folder}/frame{frame_count}.jpg", frame)
            frame_count += 1
            pbar.update(1)

    video.release()
    end = time.time()
    print(f"Extraction Complete. Total time: {int(end - start)} s. Output folder: {output_folder}")


# Define a function to crop the extracted images.
def crop_frames(x_min, x_max, y_min, y_max):
    start = time.time()

    input_dir = 'frames'
    output_dir = 'cropped_frames'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    total_images = len(os.listdir(input_dir))

    with tqdm.tqdm(total=total_images, ncols=90, desc='Cropping...', unit='images') as pbar:
        for filename in os.listdir(input_dir):
            img = cv2.imread(os.path.join(input_dir, filename))
            cropped = img[y_min:y_max, x_min:x_max]
            cv2.imwrite(os.path.join(output_dir, filename), cropped)
            pbar.update(1)

    end = time.time()
    print(f"Crop Complete. Total time: {int(end - start)} s. Output folder: {output_dir}")


# Define a Canny Detector to detect the edge.
def cannyDetector():
    start = time.time()

    source_dir = 'cropped_frames'
    output_dir = 'cannyDetection'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    total_images = len(os.listdir(source_dir))

    with tqdm.tqdm(total=total_images, ncols=90, desc='Operating...', unit='images') as pbar:
        for frame in os.listdir(source_dir):
            img = cv2.imread(os.path.join(source_dir, frame))
            # Turn the grayscale image into binary image.
            _, thresh = cv2.threshold(img, 110, 200, cv2.THRESH_BINARY)
            # Apply a morphological operation to smooth the edge.
            kernel = np.ones((5, 5), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            # Apply the canny edge detector.
            edges = cv2.Canny(opening, 10, 50)

            output_file_name = os.path.join(output_dir, frame)
            cv2.imwrite(output_file_name, edges)
            pbar.update(1)

    end = time.time()
    print(f"Canny Edge Detection Complete. Total time: {int(end - start)} s")


# Remove the bottom half of the image.
def removeBottom(below_which):
    start = time.time()

    source_dir = 'cannyDetection'
    y = int(below_which)
    total_images = len(os.listdir(source_dir))

    with tqdm.tqdm(total=total_images, ncols=90, desc='Removing...', unit='images') as pbar:
        for cannyImage in os.listdir(source_dir):
            img = cv2.imread(os.path.join(source_dir, cannyImage))
            img[y:] = 0
            cv2.imwrite(os.path.join(source_dir, cannyImage), img)
            pbar.update(1)

    end = time.time()
    print(f"Removing Complete. Total time: {int(end - start)} s. All images in {source_dir} has been operated.")


def waterLevelSearch(image):
    first_white_pixels = []
    width, height = image.shape[1], image.shape[0]

    for i in range(width):
        # 从下到上查找第一个白色像素点
        for j in range(height - 1, -1, -1):
            if image[j, i][0] == 255:
                first_white_pixels.append(j)
                break

    mid_pixel_height = int(statistics.median(first_white_pixels))

    return mid_pixel_height


def waterLevelMark(image, pixel):
    center = (int(image.shape[1] / 2), pixel)
    radius = 3
    marked = cv2.circle(image, center, radius, (0, 0, 255), 2)

    return marked


# Locate the water level height
def heightLocation(video_path):
    start = time.time()

    input_dir = 'cannyDetection'
    output_dir = 'markedImages'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    heightIndex = []
    total_images = len(os.listdir(input_dir))

    with tqdm.tqdm(total=total_images, ncols=90, desc='Searching...', unit='images') as pbar:
        for image in os.listdir(input_dir):
            img = cv2.imread(os.path.join(input_dir, image))

            mid_pixel_height = waterLevelSearch(img)
            heightIndex.append(mid_pixel_height)

            originalImage = cv2.imread(os.path.join('cropped_frames', image))
            markedImage = waterLevelMark(originalImage, mid_pixel_height)
            cv2.imwrite(os.path.join(output_dir, image), markedImage)

            pbar.update(1)

    data_path = video_path[:-4] + '_data.csv'
    pixel_locations = pd.DataFrame(heightIndex, columns=['PixelLocation'])
    pixel_locations.to_csv(data_path, index=True)

    height_rates = []

    with tqdm.tqdm(total=total_images, ncols=90, desc='Calculating...', unit='data') as pbar:
        for i in range(len(heightIndex)):
            height_rate = 1 - (heightIndex[i] / 1010)
            height_rates.append(height_rate)
            pbar.update(1)

    data = pd.read_csv(data_path)
    height_rates_df = pd.DataFrame(height_rates, columns=['HeightRate'])
    data['HeightRate'] = height_rates_df
    data.to_csv(data_path, index=True)

    end = time.time()
    duration = start - end
    formatted_time = time.strftime("%H:%M:%S", time.gmtime(duration))

    print(f"Progress Complete. Total time: {formatted_time}. Output folder: {output_dir}. Data saved in {data_path}")
