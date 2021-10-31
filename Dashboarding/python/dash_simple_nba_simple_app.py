import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

all_teams_df = pd.read_csv('src_data/shot_dist_compiled_data_2019_20.csv')

app = dash.Dash(__name__)
server = app.server
team_names = all_teams_df.group.unique()
team_names.sort()
app.layout = html.Div([
    html.Div([dcc.Dropdown(id='group-select', options=[{'label': i, 'value': i} for i in team_names],
                           value='TOR', style={'width': '140px'})]),
    dcc.Graph('shot-dist-graph', config={'displayModeBar': False}),
    dcc.Graph('shot-dist-multi-graph', config={'displayModeBar': False}),
    ])

@app.callback(
    Output('shot-dist-graph', 'figure'),
    [Input('group-select', 'value')]
)
def update_graph(grpname):
    return px.scatter(all_teams_df[all_teams_df.group == grpname], x='min_mid', y='player', size='shots_freq', color='pl_pps')

@app.callback(
    Output('shot-dist-multi-graph', 'figure'),
    [Input('group-select', 'value')]
)
def update_multi_graph(grpname):
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
        go.Scatter(x=all_teams_df.loc[all_teams_df.group == grpname, "min_mid"],
                   y=all_teams_df.loc[all_teams_df.group == grpname, "player"],
                   line=dict(color="#743de0"),
                   name="min_mid"),
                   secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=all_teams_df.loc[all_teams_df.group == grpname, "min_start"],
                   y=all_teams_df.loc[all_teams_df.group == grpname, "player"],
                   line=dict(color="#CC5500"),
                   name="min_start"),
                   secondary_y=True,
    )
    return fig

if __name__ == '__main__':
    app.run_server(debug=False)
