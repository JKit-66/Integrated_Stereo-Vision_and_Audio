import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
import seaborn as sns
import pathlib
from IPython import display
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as snss
import time
import pandas as pd
import random

DATASET_PATH = 'C:/Users/J Kit/Desktop/Y3 Intern/Week6-AudioAugmentation-ALong/4-augmentaudio/data'

DATASET_PATH_RES = 'C:/Users/J Kit/Desktop/Y3 Intern/Week6-AudioWeightTraining_Testing/tsflow_training_result'
training_set, validation_set = tf.keras.utils.audio_dataset_from_directory(
	directory= DATASET_PATH,  #'./dattaset2' #'./testFILE',
	batch_size=25,
	validation_split=0.3,
	output_sequence_length=44100,
	seed=0,
	subset='both')
 

data_dir = pathlib.Path(DATASET_PATH)
data_dir_res = pathlib.Path(DATASET_PATH_RES)

# Extracting audio labels
label_names = np.array(training_set.class_names)
print("label names:", label_names)

def squeeze(audio, labels):
  audio = tf.squeeze(audio, axis=-1)
  return audio, labels

# Applying the function on the dataset obtained from previous step
training_set = training_set.map(squeeze, tf.data.AUTOTUNE)
validation_set = validation_set.map(squeeze, tf.data.AUTOTUNE)

# Plot the waveform
def plot_wave(waveform, label):
    plt.figure(figsize=(10, 3))
    plt.title(label)
    plt.plot(waveform)
    plt.xlim([0, 16000])
    plt.ylim([-1, 1])
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)

# Convert waveform to spectrogram
def get_spectrogram(waveform):
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    return spectrogram[..., tf.newaxis]

# Plot the spectrogram
def plot_spectrogram(spectrogram, label):
    spectrogram = np.squeeze(spectrogram, axis=-1)
    log_spec = np.log(spectrogram.T + np.finfo(float).eps)
    plt.figure(figsize=(10, 3))
    plt.title(label)
    plt.imshow(log_spec, aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Time')
    

# Creating spectrogram dataset from waveform or audio data
def get_spectrogram_dataset(dataset):
    dataset = dataset.map(lambda x, y: (get_spectrogram(x), y),num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

# Applying the function on the audio dataset
train_set = get_spectrogram_dataset(training_set)
validation_set = get_spectrogram_dataset(validation_set)

# Dividing validation set into two equal val and test set
val_set = validation_set.take(validation_set.cardinality() // 2)
test_set = validation_set.skip(validation_set.cardinality() // 2)


train_set_shape = train_set.element_spec[0].shape
val_set_shape = val_set.element_spec[0].shape
test_set_shape = test_set.element_spec[0].shape

print("Train set shape:", train_set_shape)
print("Validation set shape:", val_set_shape)
print("Testing set shape:", test_set_shape)


# Defining the model
def get_model(input_shape, num_labels):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        # Resizing the input to a square image of size 64 x 64 and normalizing it
        tf.keras.layers.Resizing(64, 64),
        tf.keras.layers.Normalization(),

        # Convolution layers followed by MaxPooling layer
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.Conv2D(128, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Flatten(),

        # Dense layer
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),

        # Softmax layer to get the label prediction
        tf.keras.layers.Dense(num_labels, activation='softmax')
    ])
    # Printing model summary
    model.summary()
    return model

# Getting input shape from the sample audio and number of classes
input_shape = next(iter(train_set))[0][0].shape
print("Input shape:", input_shape)
num_labels = len(label_names)

# Creating a model
model = get_model(input_shape, num_labels)


now = time.time()


model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'],
)

EPOCHS = 25

history = model.fit(
    train_set,
    validation_data=val_set,
    epochs=EPOCHS,
)

now2 = time.time()
print(now2-now, 'seconds')   

A_L = []
B_L = []
for epo in range(EPOCHS):
   A_L.append(random.choice([1,2,3,4,5,6,7,8,9,10,11,12,13]))
   B_L.append(random.choice(['apple', 'banana', 'cherry', 'tomato', 'pineapple', 'strawberry', 'orange']))
   
data = {
    'A': A_L,
    'B': B_L
}
df = pd.DataFrame(data)

# Add a timestamp column
df['Timestamp'] = now2-now



metrics = history.history
plt.figure(figsize=(10, 5))

# Plotting training and validation loss
plt.subplot(1, 2, 1)
plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Plotting training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history.epoch, metrics['accuracy'], metrics['val_accuracy'])
plt.legend(['accuracy', 'val_accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig(os.path.join(data_dir_res,'loss_accuracy_epoches-_-.png'))

df['Loss'] = metrics['loss']
df['val_loss'] = metrics['val_loss']
df['accuracy'] = metrics['accuracy']
df['val_accuracy'] = metrics['val_accuracy']

# Write the DataFrame to a CSV file
df.to_csv(os.path.join(data_dir_res,'output_ts_comp_time-_-.csv'), index=False)



names = ['bellphone', 'drone', 'iphone', 'jogging', 'keyboard', 'silence', 'telephone']
#names = ['jogging', 'silence', 'iphone']
# Confusion matrix NOT normalized
y_pred = np.argmax(model.predict(test_set), axis=1)
y_true = np.concatenate([y for x, y in test_set], axis=0)
cm = confusion_matrix(y_true, y_pred)

# Plotting the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show(block=False)
plt.savefig(os.path.join(data_dir_res,'plot_not_normalized-_-.png'))


cm = confusion_matrix(y_true, y_pred)
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(10, 8))
sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=names, yticklabels=names)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show(block=False)

plt.savefig(os.path.join(data_dir_res,'plot_normalized-_-.png'))


report = classification_report(y_true, y_pred)
print(report)

x = data_dir/'iphone/0.1_RIR_apply1_-_0.6_RIR_apply_-_segment_iphone-ringtone.wav_1.wav'
x = tf.io.read_file(str(x))
x, sample_rate = tf.audio.decode_wav(x, desired_channels=1, desired_samples=44100,)
x = tf.squeeze(x, axis=-1)
waveform = x
x = get_spectrogram(x)
x = x[tf.newaxis,...]

class ExportModel(tf.Module):
  def __init__(self, model):
    self.model = model

    # Accept either a string-filename or a batch of waveforms.
    # YOu could add additional signatures for a single wave, or a ragged-batch.
    self.__call__.get_concrete_function(
        x=tf.TensorSpec(shape=(), dtype=tf.string))
    self.__call__.get_concrete_function(
       x=tf.TensorSpec(shape=[None, 44100], dtype=tf.float32))


  @tf.function
  def __call__(self, x):
    # If they pass a string, load the file and decode it.
    if x.dtype == tf.string:
      x = tf.io.read_file(x)
      x, _ = tf.audio.decode_wav(x, desired_channels=1, desired_samples=44100,)
      x = tf.squeeze(x, axis=-1)
      x = x[tf.newaxis, :]

    x = get_spectrogram(x)
    result = self.model(x, training=False)

    class_ids = tf.argmax(result, axis=-1)
    class_names = tf.gather(label_names, class_ids)
    return {'predictions':result,
            'class_ids': class_ids,
            'class_names': class_names}

#'./data/mini_speech_commands'

export = ExportModel(model)
export(tf.constant(str(data_dir/'jogging/0.05_RIR_apply_-_segment_running-soundscape-200116.mp3_7001.wav')))

tf_save_dir = os.path.join(data_dir_res,'tensorflowSaved-_-')
tf_save_dir = tf_save_dir.replace("\\","/")
tf.saved_model.save(export, tf_save_dir)
imported = tf.saved_model.load(tf_save_dir)
imported(waveform[tf.newaxis, :])





