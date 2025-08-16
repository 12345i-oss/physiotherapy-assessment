#augmentation.py
import cv2
import numpy as np
import os

def low_jitter(frame):
    dx, dy = np.random.randint(-5, 6, size=2)
    return np.roll(np.roll(frame, dx, axis=1), dy, axis=0)

def high_jitter(frame):
    dx, dy = np.random.randint(-20, 21, size=2)
    return np.roll(np.roll(frame, dx, axis=1), dy, axis=0)

def low_light(frame):
    return cv2.convertScaleAbs(frame, alpha=0.6, beta=0)

def extreme_low_light(frame):
    return cv2.convertScaleAbs(frame, alpha=0.3, beta=0)

def low_resolution(frame):
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (w // 4, h // 4), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

def synthetic_shadows(frame):
    h, w = frame.shape[:2]
    mask = np.zeros_like(frame, dtype=np.uint8)
    for _ in range(np.random.randint(1, 4)):
        pt1 = (np.random.randint(0, w), np.random.randint(0, h))
        pt2 = (np.random.randint(0, w), np.random.randint(0, h))
        thickness = np.random.randint(50, 150)
        cv2.line(mask, pt1, pt2, (50, 50, 50), thickness)
    return cv2.addWeighted(frame, 1, mask, 0.5, 0)

def motion_blur(frame, size=15):
    kernel = np.zeros((size, size))
    kernel[size//2, :] = np.ones(size)
    kernel /= size
    return cv2.filter2D(frame, -1, kernel)

# Augmentation mapping
AUGMENTATIONS = {
    'low_jitter': low_jitter,
    'high_jitter': high_jitter,
    'low_light': low_light,
    'extreme_low_light': extreme_low_light,
    'low_resolution': low_resolution,
    'synthetic_shadows': synthetic_shadows,
    'motion_blur': motion_blur,
}

# Augmentation application function
def apply_augmentations_to_dataset(root_dir):
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.mp4'):
                video_path = os.path.join(subdir, file)
                cap = cv2.VideoCapture(video_path)

                if not cap.isOpened():
                    print(f"Failed to open {video_path}")
                    continue

                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')

                # Prepare writers for all augmentations
                writers = {
                    name: cv2.VideoWriter(
                        os.path.join(subdir, f"{os.path.splitext(file)[0]}_{name}.mp4"),
                        fourcc, fps, (width, height)
                    )
                    for name in AUGMENTATIONS
                }

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    for name, func in AUGMENTATIONS.items():
                        aug_frame = func(frame.copy())
                        writers[name].write(aug_frame)

                cap.release()
                for writer in writers.values():
                    writer.release()

                print(f"Processed: {video_path}")


apply_augmentations_to_dataset("/Users/aks/Desktop/thesis/PHYIO-DATASET")
