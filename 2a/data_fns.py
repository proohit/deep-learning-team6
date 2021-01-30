import numpy as np
import pandas as pd
import cv2
import math
from scipy.ndimage import rotate
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from frame_fns import preprocess_frame, get_augmented_frames

def drop_problematic_videos(training, test):
    faulty_videos = pd.read_csv(
        '../Komplexe_Videos_Task2_Selectierung_Videos.csv', sep=',')
    faulty_videos.columns = ['file_name', 'problem']

    training_index_to_drop = training.merge(
        faulty_videos, on='file_name').index
    test_index_to_drop = test.merge(faulty_videos, on='file_name').index

    clean_training = training.drop(training_index_to_drop)
    clean_test = test.drop(test_index_to_drop)
    print(training.shape)
    print(test.shape)
    return clean_training, clean_test


def get_label_for_frame(frame_number, cut_frame):
    if(frame_number >= cut_frame):
        return 0
    else:
        return 1


def calculate_frame_ratio(totalFrames, cutFrame):
    upperRatio = (totalFrames - cutFrame) / totalFrames
    lowerRatio = cutFrame / totalFrames

    return lowerRatio if upperRatio > lowerRatio else upperRatio


def get_frames_labels(data, augmentations=['rotate'], size=(128, 128), use_standardization=False):
    frames = []
    labels = []
    augmented_frames = []
    augmented_labels = []
    for file in data.iloc():
        file_path = file['file_path']
        cutframe = file['cut_frame']

        # Playing video from file:
        cap = cv2.VideoCapture(file['file_path'])
        totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_ratio = calculate_frame_ratio(totalFrames, cutframe)
        minimumFrame = math.ceil(cutframe - (frame_ratio * totalFrames))
        maximumFrame = math.floor(cutframe + (frame_ratio * totalFrames))
        currentFrame = 0
        success, frame = cap.read()

        while(success):
            if use_standardization:
                if currentFrame < minimumFrame:
                    currentFrame += 1
                    success, frame = cap.read()
                    continue
                if currentFrame > maximumFrame:
                    break

            label = get_label_for_frame(currentFrame, cutframe)
            frame = preprocess_frame(frame, size=size)

            current_augmented_frames = get_augmented_frames(
                frame, augmentations=augmentations)
            augmented_frames = augmented_frames + current_augmented_frames
            for i in current_augmented_frames:
                augmented_labels.append(label)

            frames.append(frame)
            labels.append(label)
            currentFrame += 1
            success, frame = cap.read()
        cap.release()
        cv2.destroyAllWindows()
    return (frames, labels, augmented_frames, augmented_labels)


# training = {
#     "frames": [],
#     "labels": [],
#     "augmented_frames": [],
#     "augmented_labels": []
# },
# test = {
#     "frames": [],
#     "labels": [],
#     "augmented_frames": [],
#     "augmented_labels": []
# }
def build_data(training, test, validation_split):
    print(f'Basic Training frames: {len(training["frames"])}')
    print(f'Augmented Training frames: {len(training["augmented_frames"])}')

    all_training_frames = training["frames"] + training["augmented_frames"]
    all_training_labels = training["labels"] + training["augmented_labels"]
    print(f'Training frames: {len(all_training_frames)}')

    print(f'Basic Test frames: {len(test["frames"])}')
    print(f'Augmented Test frames: {len(test["augmented_frames"])}')

    all_test_frames = test["frames"] + test["augmented_frames"]
    all_test_labels = test["labels"] + test["augmented_labels"]
    print(f'Test frames: {len(all_test_frames)}')

    x_train, x_validation, y_train, y_validation = train_test_split(
        np.array(all_training_frames),
        np.array(all_training_labels),
        test_size=validation_split,
        random_state=42
    )
    print(f'Final training frames: {x_train.shape[0]}')
    print(f'Final validation frames: {x_validation.shape[0]}')

    all_test_frames = np.array(all_test_frames)
    all_test_labels = np.array(all_test_labels)
    return {
        "training":
            {
                "x": x_train,
                "y": y_train,
                "x_validation": x_validation,
                "y_validation": y_validation
            },
        "test": {
            "x": all_test_frames,
            "y": all_test_labels
            }
    }