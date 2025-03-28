import gradio as gr
import joblib
import numpy as np

model = joblib.load('random_forest_best.pkl')


neighborhood_categories = ['0', '1', '2'] 

def predict_price(garage_area, mas_vnr_area, neighborhood_category, total_bsmt_sf, full_bath, gr_liv_area):

    neighborhood_index = neighborhood_categories.index(neighborhood_category)
    
    # Create input array
    input_data = np.array([[garage_area, mas_vnr_area, neighborhood_index, total_bsmt_sf, full_bath, gr_liv_area]])
    
    # Make prediction
    estimated_price = model.predict(input_data)[0]
    return f"Estimated price: ${estimated_price:,.2f}"

# Create Gradio interface
iface = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Number(label="Garage Area"),
        gr.Number(label="Mas Vnr Area"),
        gr.Dropdown(choices=neighborhood_categories, label="Neighborhood Category"),
        gr.Number(label="Total Bsmt SF"),
        gr.Number(label="Full Bath"),
        gr.Number(label="Gr Liv Area")
    ],
    outputs=gr.Textbox(label="Estimated Price"),
    title="House Price Prediction",
    description="Enter the details below to estimate the house price."
)

# Launch the app
iface.launch()
