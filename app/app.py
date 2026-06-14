import streamlit as st
import pandas as pd
import joblib
import os
from huggingface_hub import hf_hub_download

st.set_page_config(page_title='SuperKart Revenue Predictions Portal', layout='wide')

st.markdown("""
    <div style='background-color:#0f4c81;padding:15px;border-radius:6px;margin-bottom:20px;'>
    <h2 style='color:white;text-align:center;margin:0;'>🏬 SuperKart Enterprise Predictive Sales Forecast Engine</h2>
    <p style='color:#f5f7fa;text-align:center;margin:4px 0 0 0;'>Automated MLOps Production Inference Service Dashboard</p>
    </div>
""", unsafe_allow_html=True)

# Expected feature columns (must match training data exactly)
FEATURE_COLS = [
    'Product_Weight', 'Product_Sugar_Content', 'Product_Allocated_Area',
    'Product_Type', 'Product_MRP', 'Store_Establishment_Year',
    'Store_Size', 'Store_Location_City_Type', 'Store_Type'
]

@st.cache_resource
def load_model():
    model_repo_id = os.getenv('MODEL_REPO_ID', 'shahdadpuri/superkart-sales-model')
    try:
        path = hf_hub_download(repo_id=model_repo_id, filename='best_sales_model.joblib')
        return joblib.load(path)
    except Exception as e:
        st.error(f'Error loading model from Hub: {e}')
        return None

model = load_model()

if model is None:
    st.warning('Model unavailable. Check Hugging Face Hub connection.')
else:
    with st.form('prediction_form'):
        col1, col2, col3 = st.columns(3)
        with col1:
            product_weight = st.number_input('Product Weight Value', min_value=0.0, max_value=100.0, value=12.5)
            sugar_content  = st.selectbox('Product Sugar Classification', ['Low Sugar', 'Regular', 'No Sugar'])
            allocated_area = st.slider('Product Display Allocation Ratio', 0.000, 1.000, 0.050, step=0.001)
        with col2:
            product_type = st.selectbox('Product Category Grouping', [
                'Fruits and Vegetables', 'Snack Foods', 'Frozen Foods', 'Dairy',
                'Household', 'Baking Goods', 'Canned', 'Health and Hygiene',
                'Meat', 'Soft Drinks', 'Breads', 'Hard Drinks',
                'Others', 'Starchy Foods', 'Breakfast', 'Seafood'
            ])
            product_mrp = st.number_input('Product Maximum Retail Price ($)', min_value=0.0, max_value=2500.0, value=141.50)
            est_year    = st.number_input('Store Establishment Calendar Year', min_value=1950, max_value=2026, value=2009)
        with col3:
            store_size  = st.selectbox('Store Size Bracket Metric', ['Small', 'Medium', 'High'])
            city_type   = st.selectbox('Store Location Demographics City Type', ['Tier 1', 'Tier 2', 'Tier 3'])
            store_type  = st.selectbox('Operational Store Retail Model Configuration', [
                'Supermarket Type1', 'Supermarket Type2', 'Departmental Store', 'Food Mart'
            ])
        submitted = st.form_submit_button('🔮 Compute Expected Store Revenue Target')

    if submitted:
        # Build input DataFrame with only expected feature columns
        input_df = pd.DataFrame([{
            'Product_Weight'           : float(product_weight),
            'Product_Sugar_Content'    : str(sugar_content),
            'Product_Allocated_Area'   : float(allocated_area),
            'Product_Type'             : str(product_type),
            'Product_MRP'              : float(product_mrp),
            'Store_Establishment_Year' : int(est_year),
            'Store_Size'               : str(store_size),
            'Store_Location_City_Type' : str(city_type),
            'Store_Type'               : str(store_type)
        }])[FEATURE_COLS]  # Enforce column order — prevents index_level_0 error

        try:
            prediction = model.predict(input_df)[0]
            st.markdown('---')
            st.success('### 🎯 Revenue Forecast Generated!')
            st.metric(label='Predicted Store Total Annual Revenue Stream', value=f'${prediction:,.2f}')
            st.info(
                f'📋 Strategic Horizon Breakdown Estimates: '
                f'Expected Weekly Projection: **${prediction/52:,.2f}** | '
                f'Expected Monthly Projection: **${prediction/12:,.2f}**'
            )
        except Exception as err:
            st.error(f'Prediction failed: {err}')
