import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os

st.set_page_config(
    page_title="Potato Leaf Disease Detection"
)
hide_streamlit_style = """
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
            * { font-family: 'Poppins', sans-serif; }
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'potatoes.h5')
    model=tf.keras.models.load_model(model_path, compile=False)
    return model

with st.spinner('Model is being loaded..'):
    model=load_model()

st.write("""
         # Potato Leaf Disease Detection
         """
         )
body = "Farmers who grow potatoes are facing lot of economic losses every year because of various " \
       "diseases that can happen to a potato plant. There are two common diseases known as early " \
       "blight and late blight early blight is caused by a fungus and " \
       "late blight is caused by a specific microorganism and if a farmer can detect these " \
       "diseases early and apply appropriate treatment then it can save lot of " \
       "waste and prevent the economic loss."
st.markdown(body, unsafe_allow_html=False)

file = st.file_uploader("", type=["jpg", "png"])

def import_and_detect(image_data, model):
        size = (256, 256)
        image = ImageOps.fit(image_data, size, Image.LANCZOS)
        img = np.asarray(image)
        img_reshape = img[np.newaxis, ...]
        detection = model.predict(img_reshape)
        return detection

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, width=256)
    detection = import_and_detect(image, model)
    class_names = ['Early blight', 'Late blight', 'Healthy']
    string = "Detection : " + class_names[np.argmax(detection)]
    if class_names[np.argmax(detection)] == 'Healthy':
        st.success(string)
    else:
        st.warning(string)

html_link = """
    <a href="https://github.com/Ashwinikonda/" style="color:green;" target="_blank">Github</a><br>
    <a href="https://www.linkedin.com/in/ashwinikonda/" style="color:green;" target="_blank">LinkedIn</a>
    """

st.markdown(html_link, unsafe_allow_html=True)