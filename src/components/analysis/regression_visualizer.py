"""Regression visualizer utilities for the EV/V2G analysis workflow."""

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import plotly.colors
import pickle

from dash import Dash, dcc, html, Input, Output
from typing import List, Dict, Any, Optional

from ..regression.ridge_regressor import RidgeRegressor
from ..regression.linear_regressor import LinearRegressor
from ..regression.gbm_regressor import GradientBoostingRegressor
from ..regression.xgb_regressor import XGBRegressor


class RegressionVisualizer:
    """
    A clean class replacement of a5_visualization_1.py.
    Provides:
      - 3D surface plots
      - 2D slices
      - Unified interface for ridge/xgb/gbm
    """

    def __init__(self, scenario_data, scenario_config):
        """Initialize the instance using the supplied configuration and data.
        
        Parameters
        ----------
        scenario_data : object
            Scenario data used by this operation.
        scenario_config : object
            Scenario config used by this operation."""
        self.scenario_data = scenario_data
        self.cfg = scenario_config
        self.model_dir = scenario_data.regression_dir
        self.summary_path = scenario_data.analysis_dir / "summary.csv"

        self.df = pd.read_csv(self.summary_path, sep=";")

        self.df['plug_in_duration_sq'] = self.df['plug_in_duration'] ** 2
        self.df['n_controlled_sq'] = self.df['n_controlled'] ** 2

        # Load all models found in model_dir
        self.models = {}
        for fp in self.model_dir.glob("*.pkl"):
            with open(fp, "rb") as f:
                payload = pickle.load(f)
            model_type = payload.get("model_type", None)
            self.models[model_type] = payload
        
        # Setup Options and Styles
        self.controlled_options = [{"label": c, "value": c} for c in sorted(self.df['n_controlled'].unique())]
        self.model_options = [
            {"label": "Ridge Regression", "value": "ridge"},
            {"label": "XGBoost", "value": "xgboost"},
            {"label": "Gradient Boosting", "value": "gboost"}
        ]
        self.color_list = plotly.colors.qualitative.Plotly

        # Initialize Dash App
        self.app = self._init_dash_app()
        self._setup_callbacks()
            

    # --------------------------------------------------------------------
    # Helper: generate grid for surface
    # --------------------------------------------------------------------
    def _get_surface_grid(self, selected_model: str, axes: List[str], **kwargs: Any) -> np.ndarray:
        """Generates input features (X_grid) for prediction based on selected axes."""
        grid_x = kwargs.get("grid_x")
        nc = kwargs.get("n_controlled")

        model = self.models.get(selected_model).get("model")
        scaler = self.models.get(selected_model).get("scaler")

        # 1. 1D plot (plug_in_duration vs reduced_relevant)
        if axes == ["plug_in_duration"]:
            min_range_val = kwargs.get("min_range_val")
            grid_df = pd.DataFrame({
                        "plug_in_duration": grid_x,
                        "plug_in_duration_sq": grid_x ** 2,
                        "minimum_range": min_range_val,
                        "n_controlled": nc,  
                        "n_controlled_sq": nc ** 2
                    })
            if selected_model == "ridge":
                X_grid = scaler.transform(grid_df[['plug_in_duration', 'plug_in_duration_sq', 'minimum_range', 'n_controlled', 'n_controlled_sq']])
            else: # xgboost, gboost
                X_grid = grid_df[['plug_in_duration', 'minimum_range', 'n_controlled']]

        # 2. 1D plot (minimum_range vs reduced_relevant)
        elif axes == ["minimum_range"]:
            plug_in_val = kwargs.get("plug_in_val")
            grid_df = pd.DataFrame({
                        "plug_in_duration": plug_in_val,
                        "plug_in_duration_sq": plug_in_val ** 2,
                        "minimum_range": grid_x, # grid_x is minimum_range in this case
                        "n_controlled": nc,  
                        "n_controlled_sq": nc ** 2
                    })

            if selected_model == "ridge":
                X_grid = scaler.transform(grid_df[['plug_in_duration', 'plug_in_duration_sq', 'minimum_range', 'n_controlled', 'n_controlled_sq']])
            else: # xgboost, gboost
                X_grid = grid_df[['plug_in_duration', 'minimum_range', 'n_controlled']]
                
        # 3. 2D Surface Plot
        else: # axes == ["plug_in_duration", "minimum_range"]
            grid_y = kwargs.get("grid_y")
            grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)

            grid_df = pd.DataFrame({
                "plug_in_duration": grid_xx.ravel(),
                "plug_in_duration_sq": grid_xx.ravel() ** 2,
                "minimum_range": grid_yy.ravel(),
                "n_controlled": nc,
                "n_controlled_sq": nc ** 2,
            })
            if selected_model == "ridge":
                X_grid = scaler.transform(grid_df[['plug_in_duration', 'plug_in_duration_sq', 'minimum_range', 'n_controlled', 'n_controlled_sq']])
            else: # xgboost, gboost
                X_grid = grid_df[['plug_in_duration', 'minimum_range', 'n_controlled']]

        return X_grid
    
    def _get_regression_info(self, selected_model, dff):
        """Return regression info.
        
        Parameters
        ----------
        selected_model : object
            Selected model used by this operation.
        dff : object
            Dff used by this operation.
        
        Returns
        -------
        object
            Computed result for the selected input and execution branch.
        
        Raises
        ------
        ValueError
            If an explicit validation or processing check in this operation fails."""
        model_dict = self.models.get(selected_model)
        model = model_dict.get("model")
        scaler = model_dict.get("scaler")

        if model is None:
             return f"Error: Model {selected_model} not loaded."

        if selected_model == "ridge":

            required_cols = [
                'plug_in_duration',
                'plug_in_duration_sq',
                'minimum_range',
                'n_controlled',
                'n_controlled_sq'
            ]
            
            missing = [col for col in required_cols if col not in dff.columns]

            if missing:
                raise ValueError(f"Missing required columns in dff: {missing}")

            X = dff[required_cols]
            y = dff['reduced_relevant']

            X_scaled = scaler.fit_transform(X)
            r2 = model.score(X_scaled, y)
            coefs = model.coef_

            regression_info = f"Regression R²: {r2:.3f}\nCoefficients:\n"
            for f, c in zip(required_cols, coefs):
                regression_info += f"  {f}: {c:.4f}\n"
            regression_info += f"Intercept: {model.intercept_:.4f}\n"
        else:
            X = dff[['plug_in_duration', 'minimum_range', 'n_controlled']]
            y = dff['reduced_relevant']

            r2 = model.score(X, y)
            regression_info = f"Regression R²: {r2:.3f}\n"
        
        return regression_info

    # --------------------------------------------------------------------
    # Dash Setup (Layout and Callbacks)
    # --------------------------------------------------------------------

    def _init_dash_app(self) -> Dash:
        """Initializes and configures the Dash application layout."""
        app = Dash(__name__)
        app.layout = html.Div([
            html.H3("Interactive Regression Visualization: Peak Reduction"),

            html.Div([
                dcc.Checklist(
                    id='axis-checklist',
                    options=[
                        {"label":"Plug-in Duration (X-Axis)", "value":"plug_in_duration"},
                        {"label":"Minimum Range (Y-Axis)", "value":"minimum_range"},
                    ],
                    value=["plug_in_duration", "minimum_range"], 
                    labelStyle={"display":"inline-block","margin-right":"20px"}
                ),
                dcc.Checklist(
                id='value-toggle',
                options=[
                    {"label": "Show Actual Values", "value": "actual"},
                    {"label": "Show Predicted Values", "value": "predicted"},
                ],
                value=["actual"],
                labelStyle={"display": "inline-block", "margin-right": "20px"}
                ),
            ], style={"margin-bottom":"20px"}),

            html.Div([
                html.Label("Number of controlled EVs:"),
                dcc.Dropdown(
                    id="controlled-selector",
                    options=self.controlled_options,
                    value=[self.controlled_options[5]["value"]], # Default to 6th option
                    multi=True,
                    style={"width":"25%", "display":"inline-block", "vertical-align":"top", "margin-right":"2%"}
                ),

                html.Label("Regression method:"),
                dcc.Dropdown(
                    id="model-selector",
                    options=self.model_options,
                    value="ridge",
                    multi=False,
                    style={"width":"25%", "display":"inline-block", "vertical-align":"top", "margin-right":"2%"} 
                ),
            ], style={"padding":"10px", "display":"inline-block"}),

            html.Div(
                id="regression-info",
                style={"margin": "10px 0", "font-family": "monospace", "white-space": "pre"}
            ),

            html.Div(
                dcc.Graph(id="scatter3d",
                          style={"width":"100%", "height":"100%"},
                          config={"scrollZoom": True}),
                style={"flex": "1", "padding":"10px"})
            ],
        style={
            "display": "flex", "flex-direction": "column", "height": "100vh", "margin": "0"
        })
        return app

    def _setup_callbacks(self):
        """Sets up the main Dash callback for updating the plot and info."""
        @self.app.callback(
            Output("regression-info", "children"),
            Output("scatter3d", "figure"),
            Input("axis-checklist", "value"),
            Input("controlled-selector", "value"),
            Input("value-toggle", "value"),
            Input("model-selector", "value")
        )
        def update_figure(axes: List[str], 
                          selected_controls: List[int], 
                          value_mode: List[str], 
                          selected_model: str):
            
            """Update figure.
            
            Parameters
            ----------
            axes : List[str]
                Axes used by this operation.
            selected_controls : List[int]
                Selected controls used by this operation.
            value_mode : List[str]
                Value mode used by this operation.
            selected_model : str
                Selected model used by this operation.
            
            Returns
            -------
            object
                Result computed as ``(regression_info, fig)``."""
            dff = self.df[self.df['n_controlled'].isin(selected_controls)]
            model = self.models.get(selected_model).get("model")

            # --- 1. Get Regression Info ---
            regression_info = self._get_regression_info(selected_model, dff) if "predicted" in value_mode else ""
            
            # --- 2. Generate Figure based on selected axes ---
            if set(axes) == {"plug_in_duration"}:
                fig = self._create_1d_plot_plug_in(dff, value_mode, selected_model, model)
            elif set(axes) == {"minimum_range"}:
                fig = self._create_1d_plot_min_range(dff, value_mode, selected_model, model)
            else: # ["plug_in_duration", "minimum_range"]
                fig = self._create_3d_plot(dff, value_mode, selected_model, model)

            return regression_info, fig

    # --------------------------------------------------------------------
    # Plotting Methods (Moved from original callback logic)
    # --------------------------------------------------------------------

    def _create_1d_plot_plug_in(self, dff: pd.DataFrame, value_mode: List[str], selected_model: str, model: Any) -> go.Figure:
        """Creates a 2D plot of Plug-in Duration vs. Peak Reduction."""
        fig = go.Figure()
        
        # Actual Data
        if "actual" in value_mode:
            groups = dff.groupby(['n_controlled', 'minimum_range'])
            for idx, ((nc, sl), group) in enumerate(groups):
                group_sorted = group.sort_values('plug_in_duration')
                color = self.color_list[idx % len(self.color_list)]
                fig.add_trace(go.Scatter(
                    x=group_sorted['plug_in_duration'],
                    y=group_sorted['reduced_relevant'],
                    mode='lines+markers',
                    name=f"Actual n_controlled={nc}, minimum_range={sl}",
                    line=dict(color=color, dash='solid'),
                    marker=dict(color=color)
                ))
        
        # Predicted Line
        if "predicted" in value_mode:
            unique_groups = dff[['n_controlled', 'minimum_range']].drop_duplicates()
            grid_x = np.linspace(dff["plug_in_duration"].min(), dff["plug_in_duration"].max(), 30)

            for idx, row in unique_groups.iterrows():
                nc = row['n_controlled']
                sl = row['minimum_range']
                min_range_val = sl 
                
                X_grid = self._get_surface_grid(
                    axes=["plug_in_duration"], selected_model=selected_model,
                    grid_x=grid_x, n_controlled=nc, min_range_val=min_range_val
                )

                grid_y = model.predict(X_grid)
                color = self.color_list[int((idx) % len(self.color_list))]
                fig.add_trace(go.Scatter(
                    x=grid_x, y=grid_y, mode='lines',
                    name=f"Predicted n_controlled={nc}, minimum_range={sl}",
                    line=dict(color=color, dash='dash'),
                ))

        fig.update_layout(xaxis_title="Plug-in Duration (h/month)", yaxis_title="Peak Reduction (%)")
        return fig

    def _create_1d_plot_min_range(self, dff: pd.DataFrame, value_mode: List[str], selected_model: str, model: Any) -> go.Figure:
        """Creates a 2D plot of Minimum Range vs. Peak Reduction."""
        fig = go.Figure()

        # Actual Data
        if "actual" in value_mode:
            groups = dff.groupby(['n_controlled', 'plug_in_duration'])
            for idx, ((nc, ch), group) in enumerate(groups):
                group_sorted = group.sort_values('minimum_range')
                color = self.color_list[idx % len(self.color_list)]
                fig.add_trace(go.Scatter(
                    x=group_sorted['minimum_range'],
                    y=group_sorted['reduced_relevant'],
                    mode='lines+markers',
                    name=f"Actual n_controlled={nc}, plug_in_duration={ch}",
                    line=dict(color=color, dash='solid'),
                    marker=dict(color=color)
                ))

        # Predicted Line
        if "predicted" in value_mode:
            unique_groups = dff[['n_controlled', 'plug_in_duration']].drop_duplicates()
            grid_x = np.linspace(dff["minimum_range"].min(), dff["minimum_range"].max(), 30) # X-axis is Minimum Range

            for idx, row in unique_groups.iterrows():
                nc = row['n_controlled']
                ch = row['plug_in_duration']
                plug_in_val = ch

                X_grid = self._get_surface_grid(
                    axes=["minimum_range"], selected_model=selected_model,
                    grid_x=grid_x, n_controlled=nc, plug_in_val=plug_in_val
                )
                
                grid_y = model.predict(X_grid)
                color = self.color_list[int((idx) % len(self.color_list))]
                fig.add_trace(go.Scatter(
                    x=grid_x, y=grid_y, mode='lines',
                    name=f"Predicted n_controlled={nc}, plug_in_duration={ch}",
                    line=dict(color=color, dash='dash'),
                ))

        fig.update_layout(xaxis_title="Minimum Range (km)", yaxis_title="Peak Reduction (%)")
        return fig

    def _create_3d_plot(self, dff: pd.DataFrame, value_mode: List[str], selected_model: str, model: Any) -> go.Figure:
        """Creates a 3D scatter and surface plot."""
        fig = go.Figure()

        # Actual Data (Scatter)
        if "actual" in value_mode:
            groups = dff.groupby(['n_controlled'])
            for idx, ((nc), group) in enumerate(groups):
                group_sorted = group.sort_values('minimum_range')
                color = self.color_list[idx % len(self.color_list)]

                fig.add_trace(go.Scatter3d(
                    x=group_sorted["plug_in_duration"], 
                    y=group_sorted["minimum_range"], 
                    z=group_sorted["reduced_relevant"],
                    mode="markers",
                    marker=dict(size=4, color=color),
                    name=f"Actual Values (n_controlled={nc})",
                ))

        # Predicted Surface
        if "predicted" in value_mode:
            unique_groups = dff[['n_controlled']].drop_duplicates()
            grid_x = np.linspace(self.df["plug_in_duration"].min(), self.df["plug_in_duration"].max(), 30)
            grid_y = np.linspace(self.df["minimum_range"].min(), self.df["minimum_range"].max(), 30)
            grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)

            for idx, row in unique_groups.iterrows():
                nc = row['n_controlled']

                X_grid = self._get_surface_grid(
                    axes=["plug_in_duration", "minimum_range"], selected_model=selected_model,
                    grid_x=grid_x, grid_y=grid_y, n_controlled=nc
                )
                grid_z = model.predict(X_grid).reshape(grid_xx.shape)

                color = self.color_list[int(idx % len(self.color_list))]

                fig.add_trace(go.Surface(
                    x=grid_xx, y=grid_yy, z=grid_z, opacity=0.6,
                    name=f"Predicted Surface (n_controlled={nc})",
                    colorscale=[[0, color], [1, color]],
                    showscale=False,
                ))

        fig.update_layout(
            scene=dict(
                xaxis_title="Plug-in Duration (h/month)",
                yaxis_title="Minimum Range (km)",
                zaxis_title="Peak Reduction (%)"
            ),
            legend_title_text="Controlled EVs (n_controlled)"
        )
        return fig

    def run(self, debug: bool = True, port: int = 8050):
        """Runs the Dash application server."""
        self.app.run(debug=debug, port=port)
    
