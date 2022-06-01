import collections
import glob
from typing import Optional
import seaborn as sns
import numpy as np
import pandas as pd
import pretty_midi
import tensorflow as tf
from matplotlib import pyplot as plt


configuration = tf.compat.v1.ConfigProto(device_count={"GPU": 0})
session = tf.compat.v1.Session(config=configuration)

keys_dict = ['pitch', 'step', 'duration']


#убрать counter потом
def get_midi():
    music_list = []
    print("================FILES================")
    counter = 0
    for file in glob.glob("2018/*.midi"):
        if counter < 15:
            pm = pretty_midi.PrettyMIDI(file)
            print(counter, file)
            music_list.append(pm)
            counter += 1
    return music_list


def get_music_info(musics):
    print("\n\n================Music Info================")
    for pm in musics:
        print('Количество инструментов звучит:', len(pm.instruments))
        instrument = pm.instruments[0]
        instr_name = pretty_midi.program_to_instrument_name(instrument.program)
        print('Название интрумента:', instr_name)
        for i, note in enumerate(instrument.notes[:10]):
            note_name = pretty_midi.note_number_to_name(note.pitch)
            duration = note.end - note.start
            print(f'{i}: тон={note.pitch}, нота={note_name},'
                  f' продолжительность={duration:.4f}')


def get_notes(msc) -> pd.DataFrame:
    print("\n\n================Music Logs================")
    notes = collections.defaultdict(list)
    for music in msc:
        instrument = music.instruments[0]
        sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
        prev_start = sorted_notes[0].start
        for note in sorted_notes:
            start = note.start
            end = note.end
            notes['pitch'].append(note.pitch)
            notes['start'].append(start)
            notes['end'].append(end)
            notes['step'].append(start - prev_start)
            notes['duration'].append(end - start)
            prev_start = start
    return pd.DataFrame({name: np.array(value) for name, value in notes.items()})


def plot_music_roll(notes: pd.DataFrame, count: Optional[int] = None):
    if count:
        title = f'Количество {count} нот'
    else:
        title = f'Целый датасет'
        count = len(notes['тон'])
    plt.figure(figsize=(20,4))
    plot_pitch = np.stack([notes['pitch'], notes['pitch']], axis=0)
    plot_start_and_stop = np.stack([notes['start'], notes['end']], axis=0)
    plt.plot(
        plot_start_and_stop[:, :count], plot_pitch[:, :count], color='b', marker='')
    plt.xlabel('Время сек.')
    plt.ylabel('Продолжительность')
    _ = plt.title(title)
    plt.show()


def plot_table(notes: pd.DataFrame, drop_percentile=2.5):
    #Выводим график тона
    plt.figure(figsize=[15, 5])
    plt.subplot(1, 3, 1)
    sns.histplot(notes, x="тон", bins=20)
    plt.show()
    # Выводим график шага
    plt.subplot(1, 3, 2)
    max_step = np.percentile(notes['step'], 100 - drop_percentile)
    sns.histplot(notes, x="шаг", bins=np.linspace(0, max_step, 21))
    plt.show()
    # Выводим график продолжительности мелодии через задержку
    plt.subplot(1, 3, 3)
    max_duration = np.percentile(notes['duration'], 100 - drop_percentile)
    sns.histplot(notes, x="задержка", bins=np.linspace(0, max_duration, 21))
    plt.show()


def notes_to_music_format_midi(
  notes: pd.DataFrame,
  out_file: str,
  instrument_name: str,
  velocity: int = 100,
) -> pretty_midi.PrettyMIDI:

  pm = pretty_midi.PrettyMIDI()
  instrument = pretty_midi.Instrument(
      program=pretty_midi.instrument_name_to_program(
          instrument_name))

  first_start = 0
  for i, note in notes.iterrows():
    start = float(first_start + note['step'])
    end = float(start + note['duration'])
    note = pretty_midi.Note(
        velocity=velocity,
        pitch=int(note['pitch']),
        start=start,
        end=end,
    )
    instrument.notes.append(note)
    first_start = start
  pm.instruments.append(instrument)
  pm.write(out_file)
  print('OK')
  return pm

#Обучение?????
def test_learning(msc):
    notes = get_notes(msc)
    num_notes = len(notes)
    #print('Test number note: ', num_notes)
    train_notes = np.stack([notes[key] for key in keys_dict], axis=1)
    notes_data_set = tf.data.Dataset.from_tensor_slices(train_notes)
   #print(notes_data_set.element_spec)
    return notes_data_set, num_notes


def sequence_notes(dataset: tf.data.Dataset, seq_len : int, vocab_size = 128, ) -> tf.data.Dataset:
    seq_len = seq_len+1
    #shift -> Сдвиг, stride -> шаг между элементами, drop_remainder -> удалять окна которые будут меньше size (seq_len)
    windows = dataset.window(seq_len, shift=1, stride=1, drop_remainder=True)

    #Тут мы сглаживаем
    flt = lambda x: x.batch(seq_len, drop_remainder=True)
    sequences = windows.flat_map(flt)

    def scale_pitch_ich_note(x): #нормализуем высоту каждой ноты, чтобы не звучало
        x = x/[vocab_size, 1.0, 1.0]
        return x

    def split(sequences):
        inputs = sequences[:-1]
        dense = sequences[-1]
        labes = {key:dense[i] for i, key in enumerate(keys_dict)}

        return scale_pitch_ich_note(inputs), labes

    return sequences.map(split, num_parallel_calls=tf.data.AUTOTUNE)


#делаем функцию для ноты и задержки
def mse_handler_pitch_duration(y_true: tf.Tensor, y_pred: tf.Tensor):
    mse = (y_true - y_pred) **2
    print(mse)
    pressure = 10 * tf.maximum(-y_pred, 0.0)
    return tf.reduce_mean(mse + pressure)


def tf_model(seq_length, msc, instrument_name):
    notes = get_notes(msc)
    input_shape = (seq_length, 3)
    print(input_shape)
    learning_rate = 0.005
    inputs = tf.keras.Input(input_shape)
    x = tf.keras.layers.LSTM(128)(inputs)

    outputs = {
        'pitch': tf.keras.layers.Dense(128, name='pitch')(x),
        'step': tf.keras.layers.Dense(1, name='step')(x),
        'duration': tf.keras.layers.Dense(1, name='duration')(x),
    }

    model = tf.keras.Model(inputs, outputs)

    loss = {
        'pitch': tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True),
        'step': mse_handler_pitch_duration,
        'duration': mse_handler_pitch_duration,
    }
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        loss=loss,
        loss_weights={
            'pitch': 1.6,
            'step': 1.0,
            'duration': 1.0,
        },
        optimizer=optimizer,
    )
    model.evaluate(train_ds, return_dict=True)
    print(model.summary())
    losses = model.evaluate(train_ds, return_dict=True)
    print(losses)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='./training_checkpoints/ckpt_{epoch}',
            save_weights_only=True),
        tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=5,
            verbose=1,
            restore_best_weights=True),
    ]
    epochs = 50 #колво прогонов
    history = model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
    )
    plt.plot(history.epoch, history.history['loss'], label='total loss')
    plt.show()
    temperature = 2.0
    num_predictions = 120
    sample_notes = np.stack([notes[key] for key in keys_dict], axis=1)
    input_notes = (
            sample_notes[:seq_length] / np.array([vocab_size, 1, 1]))

    generated_notes = []
    prev_start = 0
    for _ in range(num_predictions):
        pitch, step, duration = predict_note(input_notes, model, temperature)
        start = prev_start + step
        end = start + duration
        input_note = (pitch, step, duration)
        generated_notes.append((*input_note, start, end))
        input_notes = np.delete(input_notes, 0, axis=0)
        input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
        prev_start = start

    generated_notes = pd.DataFrame(
        generated_notes, columns=(*keys_dict, 'start', 'end'))
    generated_notes.head(10)
    output_file = 'ml_music.midi'
    example_pm = notes_to_music_format_midi(
        generated_notes, out_file=output_file, instrument_name=instrument_name)


def predict_note(notes: np.ndarray, keras_model: tf.keras.Model,
                   temperature: float = 1.0) -> int:
    assert temperature > 0
    inputs = tf.expand_dims(notes, 0)
    predictions = keras_model.predict(inputs)
    pitch_logits = predictions['pitch']
    step = predictions['step']
    duration = predictions['duration']
    pitch_logits /= temperature
    pitch = tf.random.categorical(pitch_logits, num_samples=1)
    pitch = tf.squeeze(pitch, axis=-1)
    duration = tf.squeeze(duration, axis=-1)
    step = tf.squeeze(step, axis=-1)
    step = tf.maximum(0, step)
    duration = tf.maximum(0, duration)
    return int(pitch), float(step), float(duration)


if __name__ == '__main__':
    msc = get_midi()
    instrument = msc.__getitem__(0).instruments[0]
    instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
    get_music_info(msc)
    row = get_notes(msc)
    row.head()
    get_more_notes = np.vectorize(pretty_midi.note_number_to_name)
    sample_note = get_more_notes(row['pitch'])
    #plot_music_roll(row, 100)
    #plot_music_roll(row)
    #plot_table(row)
    #output_file = 'try.midi'
    #example_pm = notes_to_music_format_midi(
        #row, out_file=output_file, instrument_name=instrument_name)
    test_data_set_to_train, num_note = test_learning(msc)
    seq_length = 25
    vocab_size = 128
    seq_dataset = sequence_notes(test_data_set_to_train, seq_length, vocab_size)
    batch_size = 64
    #for seq, target in seq_dataset:
    #    print('форма последовательности: ', seq.shape)
    #    print('элементы последовательности: ', seq[0: 10])
    #    print('\nтаргет: ', target)
    buffer_size = num_note - seq_length
    train_ds = (seq_dataset
                .shuffle(buffer_size)
                .batch(batch_size, drop_remainder=True)
                .cache()
                .prefetch(tf.data.experimental.AUTOTUNE))
    tf_model(seq_length, msc, instrument_name)