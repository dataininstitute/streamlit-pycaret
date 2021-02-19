from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np

model = load_model('deployment_28042020')

def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def run():

    from PIL import Image
    image = Image.open('data in istitute logo draft.png')
    image_hospital = Image.open('hospital.jpg')

    st.image(image,use_column_width=False)

    add_selectbox = st.sidebar.selectbox(
    "Comment aimeriez-vous effectuer vos prédictions?",
    ("Online", "Batch"))

    st.sidebar.info('Cette application est créée pour prédire les frais d"hospitalisation des patients')
    st.sidebar.success('https://www.pycaret.org')
    
    st.sidebar.image(image_hospital)

    st.title("Application de prévision des frais d'assurance")

    if add_selectbox == 'Online':

        age = st.number_input('Age', min_value=1, max_value=100, value=25)
        sex = st.selectbox('Sex', ['Homme', 'Femme'])
        bmi = st.number_input('IMC', min_value=10, max_value=50, value=10)
        children = st.selectbox('Enfants', [0,1,2,3,4,5,6,7,8,9,10])
        if st.checkbox('Fumeur'):
            smoker = 'oui'
        else:
            smoker = 'non'
        region = st.selectbox('Région', ['Sud Ouest', 'Nord Ouest', 'Nord Est', 'Sud Est'])

        output=""

        input_dict = {'age' : age, 'sexe' : sex, 'imc' : bmi, 'enfant' : children, 'fummeur' : smoker, 'région' : region}
        input_df = pd.DataFrame([input_dict])

        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = '$' + str(output)

        st.success('The output is {}'.format(output))

    if add_selectbox == 'Batch':

        file_upload = st.file_uploader("Uploader votre fichier csv pour effectuer vos prédicitons", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)

if __name__ == '__main__':
    run()
