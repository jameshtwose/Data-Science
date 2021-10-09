#!/usr/bin/env python
# coding: utf-8

# # PIMA diabetes Exploratory Data Analysis using plotly

# In[1]:


import pandas as pd
# import numpy as np
from utils import apply_scaling

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "svg"


# In[3]:


df = pd.read_csv("diabetes.csv")

# In[4]:


# df.head()


# In[5]:


target = "Outcome"


# In[6]:


# df.describe()


# ## Plot the amount of rows in each side of the target

# #### Looks like the target is imbalanced so this needs to be taken into account

# In[7]:


tmp = (df[target]
       .value_counts()
       .to_frame()
       .reset_index()
       .rename(columns={"index": "Outcome_name", "Outcome": "Outcome_count"})
       )


# In[8]:

app = dash.Dash()
server = app.server
preg_names = df[target].unique()
preg_names.sort()
app.layout = html.Div(
    [
        dcc.Graph('targ-bar', config={'displayModeBar': False}),

        html.Div([dcc.Dropdown(id='group-select', options=[{'label': i, 'value': i} for i in preg_names],
                               value='0', style={'width': '140px'})]),
        dcc.Graph('preg-violin', config={'displayModeBar': True}),
        dcc.Graph('preg-strip', config={'displayModeBar': True}),
    ]
)

# fig = px.violin(df
#                 .pipe(apply_scaling, "MinMax")
#                 .melt(id_vars = target),
#                 x = "variable",
#                 y = "value",
#                 color = target,
#                )
# fig.show()

@app.callback(
    Output('preg-violin', 'figure'),
    [Input('group-select', 'value')]
)
def update_graph(grpname):
    import plotly.express as px
    return px.violin(df
                     [df[target] == grpname]
                # .pipe(apply_scaling)
                .melt(id_vars = target),
                x = "variable",
                y = "value",
                # color = target,
               )

@app.callback(
    Output('preg-strip', 'figure'),
    [Input('group-select', 'value')]
)
def update_graph_2(grpname):
    import plotly.express as px
    return px.strip(df
                     [df[target] == grpname]
                # .pipe(apply_scaling)
                .melt(id_vars = target),
                x = "variable",
                y = "value",
                # color = target,
               )

@app.callback(
    Output('targ-bar', 'figure'),
    # [Input('group-select', 'value')]
)
def update_target_graph(grpname):
    import plotly.express as px
    return px.bar(data_frame=tmp,
                  x="Outcome_name",
                  y="Outcome_count",
                  color="Outcome_name")

if __name__ == '__main__':
    app.run_server(debug=False)

# fig = px.bar(data_frame=tmp,
#       x="Outcome_name",
#       y="Outcome_count",
#             color="Outcome_name")
# fig.show()


# # In[9]:
#
# fig = px.imshow(df.T)
# fig.show()
#
#
# # In[10]:
#
# fig = px.imshow(df
#                 .pipe(apply_scaling)
#                 .T)
# fig.show()
#
#
# # In[11]:
#
# fig = px.imshow(df.corr())
# fig.show()
#
#
# # In[12]:
#
# fig = px.strip(df
#                 .pipe(apply_scaling, "MinMax")
#                 .melt(id_vars = target),
#                 x = "variable",
#                 y = "value",
#                 color = target,)
# fig.show()
#
#
# # In[13]:
#
# fig = px.violin(df
#                 .pipe(apply_scaling, "MinMax")
#                 .melt(id_vars = target),
#                 x = "variable",
#                 y = "value",
#                 color = target,
#                )
# fig.show()
