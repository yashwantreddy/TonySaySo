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
    st.text("Type in a dialog to see the chances of Tony Stark saying it!")

    input_string = st.text_input('Input Line', 'I am Iron Man.')

    prediction = model.predict([input_string])
    result = "Not Tony Stark"
    img = "/tonystark_facepalm.png"
    if prediction[0] == 1:
        result = "Tony Stark"
        img = "/Tony Stark.jpg"
    prediction_conf = model.predict_proba([input_string]).max()
    col1, col2, col3 = st.beta_columns(3)

    st.markdown('<style>.prediction{color: #DC143C; font-size: 24px; font-weight: bold}</style>', unsafe_allow_html=True)

    with col1:
        st.subheader("Prediction:")
        st.markdown('<p class="prediction">' + result + '</p>', unsafe_allow_html=True)
    with col2:
        st.subheader("Confidence:")
        st.markdown('<p class="prediction">' + "{0:.2%}".format(prediction_conf) + '</p>', unsafe_allow_html=True)
    with col3:
        st.image(Image.open(char_images_path + img), width=200)
 
def render_about_the_model():
    st.header("About the Model")


    st.subheader("Implementation Details")
    st.markdown('<p class="text">This project uses <a href="https://scikit-learn.org/stable/" target="_blank">scikit-learn</a> to implement a <a href="https://en.wikipedia.org/wiki/Naive_Bayes_classifier" target="_blank">Naive Bayes Classifier</a>.  Hyperparameter\n'
                'selection is done using cross validation (10 folds).  The model is also evaluated using\n'
                'cross validation (10 folds).  With hyperparameter selection, this results in nested cross\n'
                'validation.  Stop words, which are words that provide no value to predictions (I, you,\n'
                'the, a, an, ...), are not removed from predictions.  Hyperparameter selection showed\n'
                'better performance keeping all words rather than removing <a href="https://www.geeksforgeeks.org/removing-stop-words-nltk-python/" target="_blank">NLTK\'s list of stop words</a>.\n'
                'Word counts also transformed with\n'
                'term frequencies and inverse document frequencies using scikit-learn\'s implementation.</p>', unsafe_allow_html=True)
    st.markdown('<p class="text">To see the code for the model, see this <a href="https://github.com/yashwantreddy/TonySaySo/blob/main/DidTonySaySo.ipynb" target="_blank">Jupyter Notebook.</a> </p>', unsafe_allow_html=True)

    st.subheader("Dataset Details")
    st.markdown('<p class="text">The dataset used was created for this project by parsing the Marvel released script /\n'
                'online transcript for 18 MCU movies.</p>', unsafe_allow_html=True)
    st.markdown("<p class='text'>The dataset is available on <a href='https://www.kaggle.com/pdunton/marvel-cinematic-universe-dialogue' target=_blank'>Kaggle</a>.", unsafe_allow_html=True)

    st.subheader("Why only predict Tony Stark?")
    st.markdown("<p class='text'>While the dataset contains the dialogue for all 652 character, most of which are just\n"
            "movie extras, trying to predict a large number of characters results in such poor\n"
            "performance that the model isn't useful or fun in any way.  Tony Stark being the most \n"
            "popular and most loved Avenger amongst the other characters, I believed that it would be a fun \n"
            "project to predict if a dialog has Tony's flair in it .</p>", unsafe_allow_html=True)

    st.subheader("Other Models")
    st.markdown("<p class='text'>In this project, 2 different models were buit and compared.  One used Naive Bayes,\n"
                "while the other was a Random Forest Classifier. These 2 models were cross-validated using Stratified K-fold\n"
                "cross validation alongside Grid Search Hyperparameter tuning. The Random Forest Model has a lot of tunable \n"
                "Hyperparameters and therefore took extremely long to find the best set of hyperparameters. But the wait was in vain.\n"
                "The grid-searched Random Forest model performed worse than the grid-searched Naive Bayes model.\n", unsafe_allow_html=True)

    st.subheader("Conclusion")
    st.markdown("<p class='text'>The model does not have incredible performance, due to several factors. The number of common \n"
                "dialogs between different characters - including Tony Stark - is massive. Another reason is that this is an imbalanced\n"
                "dataset problem. There are a very few examples of Tony Stark - a lot of them being very common english words (i.e. stopwords)\n"
                "This makes it harder for the model to learn the distribution and generalize well as there are a limited number of examples that \n"
                "distinguishes Tony Stark from everyone else. \n"
                "Thank you for taking the time to check this out.", unsafe_allow_html=True)

render_interactive_prediction()
render_about_the_model()
