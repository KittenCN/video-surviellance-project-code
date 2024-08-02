import os
import cv2
import numpy as np
from keras.utils import img_to_array, load_img
from keras.layers import Conv3D, ConvLSTM2D, Conv3DTranspose
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
import subprocess
# Constants
train_path = './train'
fps = 5
video_path = './Avenue_Dataset/Avenue_Dataset/training_videos'
vol_path = './Avenue_Dataset/Avenue_Dataset/training_vol'


# Function to store images in array
def store_inarray(image_path, store_image):
    image = load_img(image_path)
    image = img_to_array(image)
    image = cv2.resize(image, (227, 227), interpolation=cv2.INTER_AREA)
    gray = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]
    store_image.append(gray)

# Process training_videos dataset
def process_videos(video_path, train_images_path, fps):
    videos = os.listdir(video_path)
    for video in videos:
        output_pattern = os.path.join(train_images_path, "%03d.jpg")
        ffmpeg_command = [
            'ffmpeg',
            '-i', os.path.join(video_path, video),
            '-r', f'1/{fps}',
            output_pattern
        ]

        # Run ffmpeg command using subprocess
        try:
            subprocess.run(ffmpeg_command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {video}: {e}")
            continue

        # Process generated images
        images = os.listdir(train_images_path)
        for image in images:
            image_path = os.path.join(train_images_path, image)
            store_inarray(image_path, store_image)

# Process training_vol dataset
def process_vol(vol_path, store_image):
    vol_images_path = os.path.join(vol_path, 'frames')
    os.makedirs(vol_images_path, exist_ok=True)

    # Assuming 'vol_path' contains images similar to 'training_videos'
    vol_videos = os.listdir(vol_path)
    for video in vol_videos:
        os.system(f'ffmpeg -i {os.path.join(vol_path, video)} -r 1/{fps} {os.path.join(vol_images_path, "%03d.jpg")}')
        images = os.listdir(vol_images_path)
        for image in images:
            image_path = os.path.join(vol_images_path, image)
            store_inarray(image_path, store_image)


# Create directories if not exist
train_images_path = os.path.join(train_path, 'frames')
os.makedirs(train_images_path, exist_ok=True)

# Process both datasets
store_image = []
fps = 0.2
process_videos(video_path, train_images_path, fps)
print(process_videos(video_path, train_images_path, fps))
process_vol(vol_path, store_image)
print(process_vol(vol_path, store_image))
# Convert to numpy array and preprocess
store_image = np.array(store_image)
print(f"Shape of store_image: {store_image.shape}")  # Debugging print

if store_image.shape[0] == 0:
    print("Error: store_image is empty. Check data collection and processing.")
else:
    a, b, c = store_image.shape
    store_image.resize(b, c, a)
    store_image = (store_image - store_image.mean()) / (store_image.std())
    store_image = np.clip(store_image, 0, 1)

    # Save store_image to file
    np.save('./train/training.npy', store_image)
    print("training.npy saved successfully.")

# Define stae_model
stae_model = Sequential()
stae_model.add(
    Conv3D(filters=128, kernel_size=(11, 11, 1), strides=(4, 4, 1), padding='valid', input_shape=(227, 227, 10, 1),
           activation='tanh'))
stae_model.add(Conv3D(filters=64, kernel_size=(5, 5, 1), strides=(2, 2, 1), padding='valid', activation='tanh'))
stae_model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', dropout=0.4, recurrent_dropout=0.3,
                          return_sequences=True))
stae_model.add(
    ConvLSTM2D(filters=32, kernel_size=(3, 3), strides=1, padding='same', dropout=0.3, return_sequences=True))
stae_model.add(
    ConvLSTM2D(filters=64, kernel_size=(3, 3), strides=1, return_sequences=True, padding='same', dropout=0.5))
stae_model.add(
    Conv3DTranspose(filters=128, kernel_size=(5, 5, 1), strides=(2, 2, 1), padding='valid', activation='tanh'))
stae_model.add(
    Conv3DTranspose(filters=1, kernel_size=(11, 11, 1), strides=(4, 4, 1), padding='valid', activation='tanh'))

stae_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Load and preprocess training data
try:
    training_data = np.load('./train/training.npy')
    frames = training_data.shape[2]
    frames = frames - frames % 10

    training_data = training_data[:, :, :frames]
    training_data = training_data.reshape(-1, 227, 227, 10)
    training_data = np.expand_dims(training_data, axis=4)
    target_data = training_data.copy()

    # Train stae_model
    epochs = 5
    batch_size = 1

    callback_save = ModelCheckpoint("saved_model.h5", monitor="mean_squared_error", save_best_only=True)
    callback_early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    stae_model.fit(training_data, target_data,
                   batch_size=batch_size,
                   epochs=epochs,
                   callbacks=[callback_save, callback_early_stopping])

    stae_model.save("saved_model.h5")

except FileNotFoundError:
    print("Error: File 'training.npy' not found. Check the file path.")
