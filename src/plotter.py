import plotly.graph_objs as go
import streamlit as st

class Plotter:
    def __init__(self, models):
        """
        Initialize the Plotter with a dictionary of models.
        
        Parameters:
        models (dict): A dictionary where keys are model names and values are model instances.
        """
        self.models = models
        self.fig = go.Figure()

    def plot(self, set_type='test', window_size=100, selected_models=None):
        """
        Plot the true and predicted values.

        Parameters:
        set_type (str): Either 'test' or 'training' to plot the respective set.
        window_size (int): Number of points to display in the plot.
        selected_models (list): List of model names to plot. If None, plot all models.
        """
        if set_type == 'test':
            y_true = self.models[list(self.models.keys())[0]].data["close"][-window_size:]
        elif set_type == 'training':
            y_true = self.models[list(self.models.keys())[0]].data["close"][:-window_size]
        else:
            raise ValueError("set_type should be either 'test' or 'training'")

        self.fig = go.Figure()


        self.fig.add_trace(go.Scatter(x=y_true.index, y=y_true, mode='lines', name='True Values'))

        if selected_models is None:
            selected_models = list(self.models.keys())

        for model_name in selected_models:
            model = self.models[model_name]
            model_predictions = model.predict(set_type=set_type)

            if set_type == 'test':
                pred_index = y_true.index
            else:
                pred_index = y_true.index[:len(model_predictions)]

            self.fig.add_trace(go.Scatter(
                x=pred_index,
                y=model_predictions,
                mode='lines',
                name=f'{model_name} Predictions'
            ))


            r2_score = model.calculate_r_squared(y_true[:len(model_predictions)], model_predictions)
            self.fig.add_annotation(
                xref="paper", yref="paper",
                x=0.5, y=0.9 - 0.05 * list(self.models.keys()).index(model_name),
                text=f"{model_name} R^2: {r2_score:.4f}",
                showarrow=False
            )

        self.fig.update_layout(
            title=f'True vs Predicted Values ({set_type.capitalize()} Set)',
            xaxis_title='Date',
            yaxis_title='Close Price',
            legend_title='Legend'
        )

    def show_plot(self):
        """
        Display the plot using Streamlit.
        """
        st.plotly_chart(self.fig)