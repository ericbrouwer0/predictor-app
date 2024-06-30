import streamlit as st
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from plotly import graph_objs as go

# Import the classes from the provided script
from src.forecaster import RidgeRegressor, LassoRegressor, PredictPriceThroughNminus1
from src.plotter import Plotter

def check_data_format(data):
    if 'close' not in data.columns:
        st.error("Uploaded data must contain a 'close' column.")
        return False
    return True

def main():
    st.title("Time Series Forecasting App")

    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'fitted' not in st.session_state:
        st.session_state.fitted = False

    uploaded_file = st.file_uploader("Upload your time series data (Excel file)", type=["xlsx"])
    if uploaded_file is not None:
        data = pd.read_excel(uploaded_file, index_col=0).sort_index()
        
        if not check_data_format(data):
            return
        
        model_options = {
            'Ridge Regression': RidgeRegressor,
            'Lasso Regression': LassoRegressor,
            'N-1 Prediction': PredictPriceThroughNminus1
        }
        
        model_choice = st.selectbox("Select a model", list(model_options.keys()))
        model_class = model_options[model_choice]
        
        # Initialize the model
        model = None
        param_grid = {}

        if model_choice in ['Ridge Regression', 'Lasso Regression']:
            alpha = st.slider("Alpha (for Lasso and Ridge)", min_value=0.01, max_value=10.0, value=1.0)
        
        if model_choice == 'Lasso Regression':
            max_iter = st.slider("Max Iterations (for Lasso)", min_value=100, max_value=10000, value=1000)
            model = model_class(data_path=uploaded_file, alpha=alpha, max_iter=max_iter)
        elif model_choice == 'Ridge Regression':
            model = model_class(data_path=uploaded_file, alpha=alpha)
        else:
            model = model_class(data_path=uploaded_file)
        
        # Store the model in session state to persist it across interactions
        if st.session_state.model is None or st.session_state.model.__class__.__name__ != model_class.__name__:
            st.session_state.model = model
            st.session_state.fitted = False

        rolling_avg_window = st.selectbox("Select a rolling average window value (0 to skip)", options=list(range(6)), index=0)
        if rolling_avg_window != 0:
            st.session_state.model.add_rolling_average(rolling_avg_window)
            st.success(f"Rolling average window of {rolling_avg_window} added to the data.")

        if st.checkbox("Gridsearch CV"):
            if model_choice in ['Ridge Regression', 'Lasso Regression']:
                param_grid['alpha'] = st.text_input("Enter alpha values (comma-separated)", value="0.01,0.1,1,10").split(',')
                param_grid['alpha'] = [float(a) for a in param_grid['alpha']]
            
            tune_rolling_avg_window = st.multiselect("Select rolling average window values for tuning", options=list(range(6)), default=[1,2,3,4])
            if tune_rolling_avg_window and 0 not in tune_rolling_avg_window:
                param_grid['rolling_avg_window'] = tune_rolling_avg_window

            if st.button("Tune Model"):
                st.session_state.model.tune_hyperparameters(param_grid)
                st.success(f"Hyperparameters tuned successfully. Best hyperparameters: {st.session_state.model.model.get_params()}")
                st.session_state.fitted = True

        scale_data = st.checkbox("Scale Data", value=False)
        
        if st.button("Fit Model"):
            st.session_state.model.fit(scale=scale_data)
            st.session_state.fitted = True
            st.success("Model fitted successfully.")
        
        if st.button("Reset"):
            st.session_state.model.reset_sets()
            st.session_state.fitted = False
            st.success("Model and data have been reset.")

        if st.session_state.fitted:
            plotter = Plotter({model_choice: st.session_state.model})
            
            set_type = st.selectbox('Select Set Type for Plotting', ['test', 'training'])
            selected_models = [model_choice]  # Since we're only plotting the selected model
            
            if st.button('Plot'):
                plotter.plot(set_type=set_type, window_size=100, selected_models=selected_models)
                plotter.show_plot()
        else:
            st.warning("Please fit the model before plotting.")

if __name__ == "__main__":
    main()
