import streamlit as st
import tensorflow as tf
import numpy as np
from keras.applications.resnet import preprocess_input
from PIL import Image
import pickle

from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from keras.layers import GlobalAveragePooling2D


IMG_ADDRESS = "https://mss-p-057-delivery.stylelabs.cloud/api/public/content/8c169f67d420413ab4f5fb8639005b65?v=3c387fe3&t=775x436"
IMG_SIZE = (224, 224)
IMAGE_NAME = "user_image_biopsy.png"
BIOPSY_LABELS = [
    "Adenosis",
    "Fobroadenoma",
    "Lobular Carcinoma",
    "Mucinous Carcinoma",
    "Papillary Carcinoma",
    "Phyllodes Tumor",
    "Tubular Adenona"
]
BIOPSY_LABELS.sort()

# session states
if "biopsy" not in st.session_state:
    st.session_state.biopsy = None

    
# functions
@st.cache_resource
def get_convext_model():

    # Download the model, valid alpha values [0.25,0.35,0.5,0.75,1]
    base_model = tf.keras.applications.ConvNeXtLarge(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    # Add average pooling to the base
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    model_frozen = Model(inputs=base_model.input,outputs=x)

    return model_frozen


@st.cache_resource
def load_sklearn_models(model_path):

    with open(model_path, 'rb') as model_file:
        final_model = pickle.load(model_file)

    return final_model


def featurization(image_path, model):

    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    predictions = model.predict(img_preprocessed)

    return predictions



if not st.session_state.biopsy:
    st.error("Please Interact with Ultrasound Image First", icon="🚨")
else:
    # get the featurization model
    convext_featurized_model = get_convext_model()
    # load biopsy model
    biopsy_model = load_sklearn_models("biopsy_best_featurizor_model_no_carcinoma_CORRECT")
    # title
    st.title("Breast Cancer Classification - Biopsy Images")
    # image
    st.image(IMG_ADDRESS, caption = "Breast Cancer Classification")

    # input image
    st.subheader("Please Upload an Ultrasound Image")

    # file uploader
    image = st.file_uploader("Please Upload an Ultrasound Image", type = ["jpg", "png", "jpeg"], accept_multiple_files = False, help = "Uploade an Image")

    if image:
        user_image = Image.open(image)
        # save the image to set the path
        user_image.save(IMAGE_NAME)
        # set the user image
        st.image(user_image, caption = "User Uploaded Image")

        #get the features
        with st.spinner("Processing......."):
            image_features = featurization(IMAGE_NAME, convext_featurized_model)
            model_predict = biopsy_model.predict(image_features)
            model_predict_proba = biopsy_model.predict_proba(image_features)
            probability = model_predict_proba[0][model_predict[0]]
        col1, col2 = st.columns(2)

        with col1:
            st.header("Cancer Type")
            st.subheader("{}".format(BIOPSY_LABELS[model_predict[0]]))
        with col2:
            st.header("Prediction Probability")
            st.subheader("{}".format(probability))