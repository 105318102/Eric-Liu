import random
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, Activation, BatchNormalization, GlobalAveragePooling2D, GlobalMaxPool2D, concatenate, Dense, Dropout
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.utils import to_categorical


def train_generator(train_batch_size):
    while True:
        this_train = train_df.groupby('label_id').apply(lambda x: x.sample(n = 2000))
        shuffled_ids = random.sample(range(this_train.shape[0]), this_train.shape[0])
        for start in range(0, len(shuffled_ids), train_batch_size):
            x_batch = []
            y_batch = []
            end = min(start + train_batch_size, len(shuffled_ids))
            i_train_batch = shuffled_ids[start:end]
            for i in i_train_batch:
                x_batch.append(process_wav_file(this_train.wav_file.values[i]))
                y_batch.append(this_train.label_id.values[i])
            x_batch = np.array(x_batch)
            y_batch = to_categorical(y_batch, num_classes = len(POSSIBLE_LABELS))
            yield x_batch, y_batch


def valid_generator(val_batch_size):
    while True:
        ids = list(range(valid_df.shape[0]))
        for start in range(0, len(ids), val_batch_size):
            x_batch = []
            y_batch = []
            end = min(start + val_batch_size, len(ids))
            i_val_batch = ids[start:end]
            for i in i_val_batch:
                x_batch.append(process_wav_file(valid_df.wav_file.values[i]))
                y_batch.append(valid_df.label_id.values[i])
            x_batch = np.array(x_batch)
            y_batch = to_categorical(y_batch, num_classes = len(POSSIBLE_LABELS))
            yield x_batch, y_batch


x_in = Input(shape = (257,98,2))
x = BatchNormalization()(x_in)
for i in range(4):
    x = Conv2D(16*(2 ** i), (3,3))(x)
    x = Activation('elu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
x = Conv2D(128, (1,1))(x)
x_branch_1 = GlobalAveragePooling2D()(x)
x_branch_2 = GlobalMaxPool2D()(x)
x = concatenate([x_branch_1, x_branch_2])
x = Dense(256, activation = 'relu')(x)
x = Dropout(0.5)(x)
x = Dense(len(POSSIBLE_LABELS), activation = 'sigmoid')(x)
model = Model(inputs = x_in, outputs = x)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

from keras_tqdm import TQDMNotebookCallback
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


callbacks = [EarlyStopping(monitor='val_loss',
                           patience=5,
                           verbose=1,
                           min_delta=0.01,
                           mode='min'),
             ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=3,
                               verbose=1,
                               epsilon=0.01,
                               mode='min'),
             ModelCheckpoint(monitor='val_loss',
                             filepath='train/starter.hdf5',
                             save_best_only=True,
                             save_weights_only=True,
                             mode='min'),
             TQDMNotebookCallback()]


history = model.fit_generator(generator=train_generator(64),
                              steps_per_epoch=200,
                              epochs=20,
                              verbose=1,
                              callbacks=callbacks,
                              validation_data=valid_generator(64),
                              validation_steps=int(np.ceil(valid_df.shape[0]/64)))


model.load_weights('./train/starter.hdf5')


test_paths = glob(os.path.join('./test/', 'test/*wav'))


def test_generator(test_batch_size):
    while True:
        for start in range(0, len(test_paths), test_batch_size):
            x_batch = []
            end = min(start + test_batch_size, len(test_paths))
            this_paths = test_paths[start:end]
            for x in this_paths:
                x_batch.append(process_wav_file(x))
            x_batch = np.array(x_batch)
            yield x_batch

predictions = model.predict_generator(test_generator(50), int(np.ceil(len(test_paths)/50)))


classes = np.argmax(predictions, axis=1)








