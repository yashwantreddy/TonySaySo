import altair as alt
import pandas as pd
import streamlit as st
from joblib import load
from PIL import Image

# mcu = pd.read_pickle("svd_df.pkl")
#
# c = alt.Chart(mcu).mark_circle().encode(x='latent_1', y='latent_2', color='Tony', tooltip='line').properties(width=1300,height=900).interactive()
#
# st.write(c)

file_path = "final_models/"
char_images_path = "tony_images/"


model = load(file_path + '/main_model.joblib')

def render_interactive_prediction():

    st.header("Interactive Prediction")
    st.text("Type in a line to see which character is predicted to say it!")

    input_string = st.text_input('Input Line', 'I am Iron Man.')

    prediction = model.predict([input_string])
    result = "Not Tony Stark"
    img = "/tonystark_facepalm.png"
    if prediction[0] == 1:
        result = "Tony Stark"
        img = "/Tony Stark.jpg"
    prediction_conf = model.predict_proba([input_string]).max()
    col1, col2, col3 = st.beta_columns(3)

    st.markdown('<style>.prediction{color: red; font-size: 24px; font-weight: bold}</style>', unsafe_allow_html=True)

    with col1:
        st.subheader("Prediction:")
        st.markdown('<p class="prediction">' + result + '</p>', unsafe_allow_html=True)
    with col2:
        st.subheader("Confidence:")
        st.markdown('<p class="prediction">' + "{0:.2%}".format(prediction_conf) + '</p>', unsafe_allow_html=True)
    with col3:
        st.image(Image.open(char_images_path + img), width=200)

render_interactive_prediction()
