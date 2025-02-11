import os
import librosa
import numpy as np
from tf_keras.utils import to_categorical
from tf_keras.callbacks import EarlyStopping
from tf_keras.backend import clear_session
from sklearn.model_selection import train_test_split
from services.model import create_model

#   MODEL TRAINED SUCCESFULY, NOT AS ACCURATE AS DESIRED, SO NEEDS TO BE IMPROVED

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

def extract_features(file):
    audio, sr = librosa.load(file, duration=4, offset=0.3)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    feature_vector = np.hstack([np.mean(mfcc.T, axis=0), np.mean(chroma.T, axis=0), np.mean(contrast.T, axis=0)])
    label = int(file.split('-')[2])-1 #0-indexed #emotion_dict[int(file.split('-')[2])]
    return feature_vector, label


# def extract_features(file):
#     try:
#         audio, sr = librosa.load(file, duration=4, offset=0.3)
#         mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
#         mfcc = np.mean(mfcc.T, axis=0)
#         label = int(file.split('-')[2])-1 #0-indexed #emotion_dict[int(file.split('-')[2])]

#         return mfcc, label
#     except Exception as e:
#         print(f"File {file} could not be loaded. Error: {e}")
#         return None, None

def load_data(dataset_path):
    X, y = [], []
    for file in os.listdir(dataset_path):
        file_path = os.path.join(dataset_path, file)
        f_x, f_y = extract_features(file_path)
        if f_x is not None and f_y is not None:
            X.append(f_x)
            y.append(f_y)

    return np.array(X), np.array(y)


def main():
    dataset_path = 'datasets/ravdess/'

    X, y = load_data(dataset_path)
    X = np.expand_dims(X, axis=-1)

    num_classes = len(set(y))

    y = to_categorical(y, num_classes)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    input_shape = (X.shape[1], 1)

    model = create_model(input_shape, num_classes)

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)#, callbacks=[early_stopping])

    model.save('models/emotion_model.h5')


if __name__=='__main__':
    clear_session()
    main()