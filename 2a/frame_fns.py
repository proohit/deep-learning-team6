import cv2


def get_augmented_frames(frame, augmentations):
    augmented_frames = []
    if 'flip' in augmentations:
        flipped_frame = cv2.flip(frame, 1)
        augmented_frames.append(flipped_frame)
    if 'color' in augmentations:
        colored_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        augmented_frames.append(colored_frame)
        colored_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        augmented_frames.append(colored_frame)
    # farben anpassen
    # skalieren
    # zoom
    return augmented_frames


def preprocess_frame(frame, size, color=cv2.COLOR_BGR2RGB):
    frame = cv2.cvtColor(frame, color)
    frame = cv2.resize(frame, size)
    return frame