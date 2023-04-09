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

st.write('Mood Music')



# prompt = "Hello, OpenAI!"

# # Call the OpenAI API to generate a response
# response = openai.Completion.create(
#   engine="davinci",
#   prompt=prompt,
#   max_tokens=10,
# )

# # Print the generated text
# st.write(response.choices[0].text)





#sdfghjk

filename = 'sentiment_model.pkl'
clf = pickle.load(open(filename, 'rb'))


tokenizer = AutoTokenizer.from_pretrained('tokenizer_info')

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def preprocess_input_text(text):
    # Tokenize input text
    st.write('in')
    
    encoded_text = tokenizer.encode(text, padding=True, truncation=True, return_tensors='tf')
    encoded_text = encoded_text.numpy()
    
    st.write(encoded_text)

    # Convert encoded text back into words
    words = [tokenizer.decode([token]) for token in encoded_text[0]]
    input_text = ' '.join(words)
    st.write(input_text)
    vector = vectorizer.transform([input_text])

    return vector

try:
    user_set = st.file_uploader("upload file", type={"csv"})
    user_set = pd.read_csv(user_set)
    user_set.drop('label', axis=1, inplace=True)
    
    submit = st.button('Go')
    
    if submit:    
        analyze(user_set)  
        # st.write(user_set)
        
        key_input = (get_key(get_ma_mi(user_set)))
        st.write(key_input)
        
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

        # create MIDI file
        
        mf = midi.translate.streamToMidiFile(melody)
        
        midi_data = io.BytesIO()
        # mf.writestr(midi_data)
        st.write(midi_data)
        midi_data.seek(0)
        
        st.download_button(
            label='Download MIDI',
            data=midi_data.getvalue(),
            file_name='music.mid',
            mime='audio/midi')

        # predict
        
        st.write('loading...')
        
        for row in user_set.itertuples():
            i = row.Index
            st.write(i)
            text = row.text
            st.write(text)
            processed_text = preprocess_input_text(text)
            prediction = clf.predict(processed_text)
            user_set.loc[i, 'predictions'] = prediction
            
        # plot
        fig, ax = plt.subplots()
        sns.countplot(x='predictions', data=user_set)
        plt.title('Distribution of mental health states')
        plt.xticks([0, 1, 2, 3], ['joy', 'fear', 'anger', 'sadness'])
        plt.ylabel('Count')
        st.pyplot(fig)
      
        emotion_map = {
            0: 'joy',
            1: 'fear',
            2: 'anger',
            3: 'sadness',
        }

        user_set['predictions'] = user_set['predictions'].map(lambda x: emotion_map[x])       
        st.write(user_set)
        user_set['predictions'] = pd.Categorical(user_set['predictions'], categories=emotion_map.values())
        
        emotion = user_set['predictions'].value_counts().idxmax()
        st.write(f'Your mental state is dominated by {emotion}.')
        
        # generate prompt
        
        def generate_response(prompt):
            st.write('is this being called?')
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=30,
                n=1,
                stop=None,
                temperature=0.5,
            )
            st.write(response.choices)
            st.write(response.choices[0].text)
            return response.choices[0].text

        st.write(generate_response('i like pie!'))
        string = f"Give a caption for an image that is a metaphorical symbol of {emotion}:"
        st.write(string)
        suggested_response = generate_response(string)
        suggested_response = suggested_response.split(":")[0]
        # suggested_response = suggested_response.strip().replace("'", "")    
        
        st.write('response below')
        st.write(suggested_response)    
        
        # image gen
        response = openai.Image.create(
            prompt=suggested_response,
            n=1,
            size="512x512",
        )
        
        image_url = response['data'][0]['url']
        
        st.write(image_url)
        
        st.image(image_url, caption="Generated image")        
    
except:
    pass
