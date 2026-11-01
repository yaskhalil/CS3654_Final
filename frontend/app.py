from pathlib import Path
import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, dcc, html

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "bronze" / "flights.csv"
df = pd.read_csv(DATA_PATH)

df["arr_delay"] = pd.to_numeric(df["arr_delay"], errors="coerce")
df["dep_delay"] = pd.to_numeric(df["dep_delay"], errors="coerce")

carriers = sorted(df["carrier"]. dropna().unique())
origins = sorted(df["origin"].dropna().unique())
delay_min = int(df["arr_delay"].min())
delay_max = int(df["arr_delay"].max())


app = Dash(__name__)
app.layout = html.Div(
    [
        html.H1("Flights Analytics"),
        html.Button("How to use", id="help-button"),
        html.Div(
            "Choose filters, then explore the charts in the Explore tab.",
            id="help-text",
            style={"display": "none", "margin": "10px 0"},
        ),
        dcc.Tabs(
            id="tabs",
            value="overview",
            children=[
                dcc.Tab(label="Overview", value="overview"),
                dcc.Tab(label="Explore Flights", value="explore"),
                dcc.Tab(label="Statistics", value="statistics"),
                dcc.Tab(label="Models", value="models"),
                dcc.Tab(label="Prediction", value="prediction"),
            ],
        ),

        html.Div(
            id="shared-filters",
            children=[
                html.Label("Carrier"),
                dcc.Dropdown(
                    id="carrier-filter",
                    options=[{"label": c, "value": c} for c in carriers],
                    placeholder="All carriers",
                    multi=True,
                ),
                html.Label("Origin"),
                dcc.Dropdown(
                    id="origin-filter",
                    options=[{"label": a, "value": a} for a in origins],
                    placeholder="All origins",
                    multi=True,
                ),
                html.Label("Arrival delay range (minutes)"),
                dcc.RangeSlider(
                    id="arrival-delay-slider",
                    min=delay_min,
                    max=delay_max,
                    value=[max(delay_min, -30), min(delay_max, 60)],
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ],
            style={"padding": "16px", "maxWidth": "900px"},
        ),
        html.Div(id="tab-content", style={"padding": "16px"})
    ],
    style={"fontFamily": "Arial", "margin": "24px"},
)

@app.callback(
    Output("help-text", "style"),
    Input("help-button", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_help(n_clicks):
    if n_clicks and n_clicks % 2:
        return {"display": "block", "margin": "10px 0"}
    return {"display": "none"}

@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value"),
    Input("carrier-filter", "value"),
    Input("origin-filter", "value"),
    Input("delay-filer-slider", "value"),
)
def update_tab(tab, selected_carriers, selected_origins, delay_range):
    filtered = df.dropna(subset=["arr_delay"]).copy()
    if selected_carriers:
        filtered = filtered[filtered["carrier"].isin(selected_carriers)]
    if selected_origins:
        filtered = filtered[filtered["origin"].isin(selected_origins)]
    if delay_range:
        filtered = filtered[filtered["arr_delay"].between(delay_range[0], delay_range[1])]

    if tab == "overview":
        return html.Div(
            [
                html.H2("Overview"),
                html.P(f"Flights matching filters: {len(filtered):,}"),
                html.P(
                    "Average arrival delay: "
                    f"{filtered['arr_delay'].mean():.1f} minutes"
                    if not filtered.empty
                    else "No flights match these filters."
                ),
            ]
        )
    if tab == "explore":
        if filtered.empty:
            return html.P("No flights match these filters.")

        by_carrier = (
            filtered.groupby("carrier", as_index=False)["arr_delay"]
            .mean()
            .sort_values("arr_delay")
        )
        delay_plot = px.histogram(
            filtered,
            x="arr_delay",
            nbins=40,
            title="Arrival delay distribution",
        )
        carrier_plot = px.bar(
            by_carrier,
            x="carrier",
            y="arr_delay",
            title="Average arrival delay by carrier",
        )

        return html.Div(
            [
                dcc.Graph(figure=delay_plot),
                dcc.Graph(figure=carrier_plot),
            ]
        )
    if tab == "statistics":
        return html.Div(
            [
                html.H2("Statistics"),
                html.P(f"Flights in filtered data: {len(filtered):,}"),
                html.P(
                    f"Median arrival delay: {filtered['arr_delay'].median():.1f} minutes"
                    if not filtered.empty
                    else "No flights match these filters."
                ),
                html.P("Add your precomputed normality test results here."),
            ]
        )


    if tab == "models":
        return html.Div(
            [
                html.H2("Models"),
                html.P("Add saved PCA, DBSCAN, and model evaluation results here."),
            ]
        )

    if tab == "prediction":
        return html.Div(
            [
                html.H2("Prediction"),
                html.P(
                    "Add inputs for a future flight here, then load your saved "
                    "MLP model to predict its delay."
                ),
            ]
        )

    return html.P("Select a dashboard tab.")

if __name__ == "__main__":
    app.run(debug=True)