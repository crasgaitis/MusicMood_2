from utils import get_key, get_ma_mi, analyze
import streamlit as st 
import openai
import pickle
from music21 import *
import random
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import io
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os
# from configure import API_KEY
import tensorflow as tf
from transformers import AutoTokenizer

# os.environ["OPENAI_API_KEY"] = API_KEY
# openai.api_key = os.environ["OPENAI_API_KEY"]

os.environ["OPENAI_API_KEY"] = st.secrets['APIKEY']
openai.api_key = os.environ["OPENAI_API_KEY"]

st.image("https://cdn.discordapp.com/attachments/1021852803905359984/1094550872094146620/final_music_mood_logo.jpg")
st.header('Music Mood')
st.markdown('Making a *safer* mental health space, powered by AI.')

st.markdown(':orange[Upload your message history.]')

#sdfghjk

filename = 'sentiment_model.pkl'
clf = pickle.load(open(filename, 'rb'))


tokenizer = AutoTokenizer.from_pretrained('tokenizer_info')

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def preprocess_input_text(text):
    # Tokenize input text

    encoded_text = tokenizer.encode(text, padding=True, truncation=True, return_tensors='tf')
    encoded_text = encoded_text.numpy()

    # Convert encoded text back into words
    words = [tokenizer.decode([token]) for token in encoded_text[0]]
    input_text = ' '.join(words)

    vector = vectorizer.transform([input_text])

    return vector

try:
    user_set = st.file_uploader("upload file", type={"csv"})
    user_set = pd.read_csv(user_set)
    # user_set.drop('label', axis=1, inplace=True)
    
    st.markdown(':orange[Get an artistic rendering of your mental health state, with audio and visuals!]')
    submit = st.button('Submit')
    
    if submit:    
        analyze(user_set)  
        # st.write(user_set)
        
        key_input = (get_key(get_ma_mi(user_set)))
        
        key_str = key_input
        key_obj = key.Key(key_str)
        time_signature = meter.TimeSignature('4/4')

        # melody: random notes
        melody = stream.Stream()
        melody.append(key_obj)
        melody.append(time_signature)

        for i in range(8):
            note_name = random.choice(scale.MajorScale(key_str).getPitches() + scale.MelodicMinorScale(key_str).getPitches())
            note_obj = note.Note(note_name)
            note_obj.duration = duration.Duration(random.choice([0.25, 0.5, 1, 2]))
            melody.append(note_obj)

        # play
        # melody.show('midi')

        # create MIDI file
               
        mf = midi.translate.streamToMidiFile(melody)
        
        midi_data = io.BytesIO()
        # mf.writestr(midi_data)
        st.write(midi_data)
        midi_data.seek(0)
        
        st.download_button(
            label='🎵 Download MIDI 🎵',
            data=midi_data.getvalue(),
            file_name='music.mid',
            mime='audio/midi')

        # predict
        
        st.write('loading...')
        
        for row in user_set.itertuples():
            i = row.Index
            text = row.text
            processed_text = preprocess_input_text(text)
            prediction = clf.predict(processed_text)
            user_set.loc[i, 'predictions'] = prediction
            
        # plot
        fig, ax = plt.subplots()
        sns.countplot(x='predictions', data=user_set, palette = ['#EB54E5', '#5EB964', '#72ADEE', '#FEF258'])
        plt.title('Distribution of mental health states')
        plt.xticks([0, 1, 2, 3], ['joy', 'fear', 'anger', 'sadness'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.ylabel('Count')
        plt.xlabel('')
        st.pyplot(fig)
      
        emotion_map = {
            0: 'joy',
            1: 'fear',
            2: 'anger',
            3: 'sadness',
        }
        
        st.write('See your mental health analysis message-by-message:')

        user_set['predictions'] = user_set['predictions'].map(lambda x: emotion_map[x])       
        st.dataframe(user_set, height = 150)
        user_set['predictions'] = pd.Categorical(user_set['predictions'], categories=emotion_map.values())
        
        emotion = user_set['predictions'].value_counts().idxmax()
        st.subheader(f'Your mental state is dominated by {emotion}.')
        
        # generate prompt
        
        def generate_response(prompt):
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=30,
                n=1,
                stop=None,
                temperature=0.6,
            )
            
            return response.choices[0].text

        string = f"Give a caption for an image that is a creative metaphorical symbol of {emotion}:"
        suggested_response = generate_response(string)
        suggested_response = suggested_response.split(":")[0]
        # suggested_response = suggested_response.strip().replace("'", "")    
        
        st.write('Your mental state, visualized.')    
        
        # image gen
        response = openai.Image.create(
            prompt=suggested_response,
            n=1,
            size="512x512",
        )
        
        image_url = response['data'][0]['url']
        
        st.image(image_url, caption=suggested_response)  
        
        st.markdown(':orange[Mental health is a journey for everyone! Share your image, audio, and how you\'re doing on social media or click submit again to regenerate.]')      
    
except:
    pass
