import streamlit as st
import pandas as pd
import joblib

# Loading the trained models
cat_model = joblib.load('cat_model.joblib')
lin_pipe = joblib.load('lin_pipe.joblib')
ada_pipe = joblib.load('ada_pipe.joblib')
extra_pipe = joblib.load('extra_pipe.joblib')

# Input fields for features
category = st.selectbox('Category', ['jewelry.pendant', 'jewelry.necklace', 'jewelry.earring', 'jewelry.ring',
 'jewelry.brooch', 'jewelry.bracelet', 'jewelry.souvenir', 'jewelry.stud'])
brand_id = st.number_input('Brand ID', min_value=1, max_value=1000, value=1)
target_gender = st.selectbox('Target Gender', ['f', 'nan', 'm'])
main_color = st.selectbox('Main Color', ['white', 'red', 'yellow', 'nan', 'unknown-color'])
main_metal = st.selectbox('Main Metal', ['gold', 'silver', 'nan', 'platinum'])
main_gem = st.selectbox('Main Gem', ['sapphire', 'diamond', 'amethyst', 'nan', 'fianit', 'pearl', 'quartz', 'topaz',
 'garnet', 'quartz_smoky', 'ruby', 'agate', 'mix', 'citrine', 'emerald', 'amber',
 'chrysolite', 'chrysoprase', 'nanocrystal', 'turquoise', 'sitall',
 'corundum_synthetic', 'coral', 'onyx', 'nacre', 'spinel', 'tourmaline',
 'emerald_geothermal', 'garnet_synthetic', 'rhodolite', 'sapphire_geothermal'])

# Model selection dropdown
model_choice = st.selectbox('Select Model', ['CatBoost', 'Linear Regression', 'AdaBoost', 'ExtraTrees'])

# Create a dataframe from inputs
input_data = pd.DataFrame({
    'Category': [category],
    'Brand_ID': [brand_id],
    'Target_Gender': [target_gender],
    'Main_Color': [main_color],
    'Main_Metal': [main_metal],
    'Main_Gem': [main_gem]
})

# Make prediction when button is clicked
if st.button('Predict Price'):
    if model_choice == 'CatBoost':
        prediction = cat_model.predict(input_data)
        st.success(f'The predicted price using CatBoost is: ${prediction[0]:.2f}')
    elif model_choice == 'Linear Regression':
        prediction = lin_model.predict(input_data)
        st.success(f'The predicted price using Linear Regression is: ${prediction[0]:.2f}')
    elif model_choice == 'AdaBoost':
        prediction = ada_model.predict(input_data)
        st.success(f'The predicted price using AdaBoost is: ${prediction[0]:.2f}')
    elif model_choice == 'ExtraTrees':
        prediction = extra_model.predict(input_data)
        st.success(f'The predicted price using ExtraTrees is: ${prediction[0]:.2f}')





# Input fields for features
category = st.selectbox('Category', ['jewelry.pendant', 'jewelry.necklace', 'jewelry.earring', 'jewelry.ring',
 'jewelry.brooch', 'jewelry.bracelet', 'jewelry.souvenir', 'jewelry.stud'])
brand_id = st.number_input('Brand ID', min_value=1, max_value=1000, value=1)
target_gender = st.selectbox('Target Gender', ['f', 'nan', 'm'])
main_color = st.selectbox('Main Color', ['white', 'red', 'yellow', 'nan', 'unknown-color'])
main_metal = st.selectbox('Main Metal', ['gold', 'silver', 'nan', 'platinum'])
main_gem = st.selectbox('Main Gem', ['sapphire', 'diamond', 'amethyst', 'nan', 'fianit', 'pearl', 'quartz', 'topaz',
 'garnet', 'quartz_smoky', 'ruby', 'agate', 'mix', 'citrine', 'emerald', 'amber',
 'chrysolite', 'chrysoprase', 'nanocrystal', 'turquoise', 'sitall',
 'corundum_synthetic', 'coral', 'onyx', 'nacre', 'spinel', 'tourmaline',
 'emerald_geothermal', 'garnet_synthetic', 'rhodolite', 'sapphire_geothermal'])

# Create a dataframe from inputs
input_data = pd.DataFrame({
    'Category': [category],
    'Brand_ID': [brand_id],
    'Target_Gender': [target_gender],
    'Main_Color': [main_color],
    'Main_Metal': [main_metal],
    'Main_Gem': [main_gem]
})



# Make prediction when button is clicked
if st.button('Predict Price'):
    prediction = cat_model.predict(input_data)
    st.success(f'The predicted price is: ${prediction[0]:.2f}')

