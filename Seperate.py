import streamlit as st
import tensorflow as tf
import librosa
import io
import pandas as pd
import numpy as np
from keras.models import load_model
import tensorflow_io as tfio

gen = load_model('waveunet_Dhimey.h5')
gen1 = load_model('waveunet_Bansuri.h5')
gen2 = load_model('waveunet_Vocal.h5')

from scipy.io.wavfile import write
def seperated_auido_gan(audio, model):
    tensor = tf.convert_to_tensor(audio)
    mixture_spectogram = tfio.audio.spectrogram(tensor, nfft = 1022, window = 1022, stride=256)
    number_of_chunks = mixture_spectogram.shape[0]//256
    target = np.zeros([mixture_spectogram.shape[0],512])
    for i in range(number_of_chunks):
        START = i*256
        END = START + 256

        S_mix_new=mixture_spectogram[START:END, :]

        X=tf.reshape(S_mix_new, (1, 256, 512, 1))
        y=model.predict(X, batch_size=32)
        target[START:END,:] = y.reshape(256,512)
    S_mix_new=mixture_spectogram[-256:, :]
    X=tf.reshape(S_mix_new,(1, 256, 512, 1))
    y=model.predict(X, batch_size=32)
    target[-256:,:] = y.reshape(256,512)
    y = tfio.audio.inverse_spectrogram(target.astype(np.float32),nfft=1022, window=1022, stride=256, iterations = 50)
    return y

def seperated_auido_unet(audio, model):
    y_mix = librosa.core.resample(audio,orig_sr=44100,target_sr=44100)
    S_mix = librosa.stft(y_mix,n_fft=1024, hop_length=512)
    S_mix = librosa.amplitude_to_db(np.abs(S_mix))


    number_of_chunks = S_mix.shape[1]//128
    target = np.zeros([512,S_mix.shape[1]])
    for i in range(number_of_chunks):
        START = i*128
        END = START + 128

        S_mix_new=S_mix[:, START:END]

        X=S_mix_new[1:].reshape(1, 512, 128, 1)

        y=model.predict(X, batch_size=32)
        target[:,START:END] = y.reshape(512,128)
    S_mix_new=S_mix[:, -128:]
    X=S_mix_new[1:].reshape(1, 512, 128, 1)

    y=model.predict(X, batch_size=32)
    target[:,-128:] = y.reshape(512,128)

    D_reverse = librosa.db_to_amplitude(target)
    y = librosa.griffinlim(D_reverse, hop_length=512, n_iter=100)
    return y
# mixture_spectogram = tfio.audio.spectrogram(mixture_audio, nfft = 1022, window = 1022, stride=256)

# Function to convert audio tensor to playable format and save to file
def convert_and_save_audio(audio_tensor, output_file='output.wav', sample_rate=44100):
    # Reshape the flattened audio tensor
    audio_data = np.reshape(audio_tensor, (-1, 1))  # Assuming stereo audio, adjust shape accordingly

    # Normalize audio data if needed
    audio_data = audio_data / np.max(np.abs(audio_data))

    # Convert to 16-bit integer
    audio_data = (audio_data * 32767).astype(np.int16)

    # Write audio data to WAV file
    write(output_file, sample_rate, audio_data)

    return output_file
 
def div_chunk(audio_data):
    x_list1 = []
    # ### Trim the audio_data to make it a multiple of chunk_size
    num_elements_to_keep = len(audio_data) // 64 * 64
    audio_data = audio_data[:num_elements_to_keep]

    # split the audio and vocals data into 64-sample chunks
    audio_chunks = [audio_data[i:i+64] for i in range(0, len(audio_data),64)]


    # add the chunks to the x_list and y_list
    x_list1.extend(audio_chunks)

    # create the DataFrame with x and y as columns
    df_pred = pd.DataFrame({'x': x_list1})
    x_test = tf.expand_dims((tf.constant(df_pred['x'].apply(lambda x: list(x)).tolist())),0)
    return x_test
# Title
st.title('Music Source Seperation')

# Upload audio file
audio_file = st.file_uploader("Upload audio file", type=["mp3", "wav", "ogg", "flac","m4a"])

# Check if audio file is uploaded
if audio_file is not None:
    # Read the audio file
        # Display audio player
    st.audio(audio_file, format='audio/wav')
    audio_bytes = audio_file.read()
    

    # Load audio data
    audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)


    audio_np = np.array(audio)
    chunk=div_chunk(audio_np)

    col1, col2, col3= st.columns(3)

    with col1:
        button1 = st.button('Dhime')

    with col2:
        button2 = st.button('Bansuri')

    with col3:
        button3 = st.button('Vocal')
    
    if button1:
        st.text('Wave UNET')
        y_pred= gen.predict(chunk)
                # Convert and save audio
        output_file = convert_and_save_audio(y_pred.flatten())
        # Display the audio player in Streamlit
        st.audio(output_file, format='audio/wav')
        st.text('UNET using Spectrogram')
        model = load_model('UNET_Dhimey.h5')
        audio_unet = seperated_auido_unet(audio, model)
        output_file = convert_and_save_audio(audio_unet.flatten())
        st.audio(output_file, format='audio/wav')
        st.text('UNET GAN')
        model = load_model('GAN_Dhimey.h5')
        audio_unet = seperated_auido_gan(audio, model)
        output_file = convert_and_save_audio(audio_unet)
        st.audio(output_file, format='audio/wav')

    if button2:
        st.text('Wave UNET')  
        y_pred= gen1.predict(chunk)
        # Convert and save audio
        output_file = convert_and_save_audio(y_pred.flatten())
        # Display the audio player in Streamlit
        st.audio(output_file, format='audio/wav')
        st.text('UNET using Spectrogram')
        model = load_model('UNET_Bansuri.h5')
        audio_unet = seperated_auido_unet(audio, model)
        output_file = convert_and_save_audio(audio_unet.flatten())
        st.audio(output_file, format='audio/wav')
        st.text('UNET GAN')
        model = load_model('GAN_Bansuri.h5')
        audio_unet = seperated_auido_gan(audio, model)
        output_file = convert_and_save_audio(audio_unet)
        st.audio(output_file, format='audio/wav')

    if button3:  
        st.text('Wave UNET') 
        y_pred= gen2.predict(chunk)
        # Convert and save audio
        output_file = convert_and_save_audio(y_pred.flatten())
        # Display the audio player in Streamlit
        st.audio(output_file, format='audio/wav')
        st.text('UNET using Spectrogram')
        model = load_model('UNET_Vocals.h5')
        audio_unet = seperated_auido_unet(audio, model)
        output_file = convert_and_save_audio(audio_unet.flatten())
        st.audio(output_file, format='audio/wav')
        st.text('UNET GAN')
        model = load_model('GAN_Vocal.h5')
        audio_unet = seperated_auido_gan(audio, model)
        output_file = convert_and_save_audio(audio_unet)
        st.audio(output_file, format='audio/wav')


