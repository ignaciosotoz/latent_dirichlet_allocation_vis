#!/usr/bin/env python3

import dash
import json
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_objs as gfx
import plotly.express as px
import plotly.figure_factory as fig_fact
import dash_table
import pandas as pd
import joblib
from train import CreateLDAModel
from helpers import  DiagnosisLDA


# Prep trained model
# trained_model = DiagnosisLDA(original_data = pd.read_csv('pickles/concatenated_data.csv'),
                             # model = joblib.load('pickles/lda_10_topics_train.pkl'),
                             # features = joblib.load('pickles/lda_10_topics_features.pkl'),
                             # counter = joblib.load('pickles/lda_10_topics_counter.pkl'))

trained_model = DiagnosisLDA(original_data=pd.read_csv('pickles/concatenated_data.csv'),
                             model = joblib.load('pickles/lda_5_train.pkl'),
                             features = joblib.load('pickles/lda_5_features.pkl'),
                             counter = joblib.load('pickles/lda_5_counter.pkl'))

# extract concatenated train data   
data = trained_model.concatenated_sources
# get inferred topics (call method)
trained_model.infer_topics_on_model()
# get probability mixture (call method)
trained_model.infer_probability_mixture()
labeled_data = trained_model.transformed_features_via_lda



get_columns_for_data_table = lambda x: [{'name': i, "id": i, "deletable": False} for i in x.columns]
get_column_property = [{"name": i, "id": i, "deletable": True} for i in trained_model.topics]
tidy_button_pairing = lambda x: [{'label': i, 'value': i} for i in x]
words_per_topic = pd.DataFrame(trained_model.topics.values()).T
words_per_topic.columns = list(map(lambda x: f"Tópico {x}", list(trained_model.topics.keys())))
# words_per_topic = words_per_topic.T
data['lyrics'] = data['lyrics'].str.slice(0, 50) + '...'
data = data.drop(columns='Unnamed: 0')
artist_relevance = data['artist'].value_counts()


def report_model(trained_model = trained_model):
    return html.P(f"Learning Decay {trained_model.best_estimator.learning_decay}"), html.P("N Tópicos: {trained_model.best_estimator.n_components}")

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1(children='Identificación de tópicos en Letras'),
            html.Div(children='Una aplicación demo montada en Dash')
        ]),
   ]),
    dbc.Row([
        html.H4('Sobre los datos'),
        html.P('Los datos corresponden todos las canciones y sus respectivas letras para una serie finita de artistas. Las letras fueron descargadas mediante la API de Genius.'),
        html.P('El objetivo es desarrollar un modelo probabilístico generativo que identifique la composición de un número finito de tópicos, dado un conjunto finito de documentos. Este modelo a implementar se conoce como Asignación Latente de Dirichlet (Latent Dirichlet Allocation, Blei, Ng & Jordan, 2003)'),
        html.P('Cantidad de registros: 10840. Cantidad de artistas: 72')
    ]),
    html.Div([
        dbc.Card([
            html.H5("Agenda de entrenamiento actual", className='card-title'),
            report_model()
        ], )
    ]),
    html.Div([
        html.H2("Composición ejemplo de la base de datos"),
        dash_table.DataTable(id='explore-lyrics-output',
                             columns = get_columns_for_data_table(data),
                             data = data.to_dict('records'),
                             filter_action="native",
                             fixed_rows = {'headers': True, 'data': 0},
                             style_cell = {'width': '150px'},
                             style_data={'whiteSpace': 'normal'},
                             css=[{
                                'selector': '.dash-cell div.dash-cell-value',
                                'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
                                }]
                             ),
            ]),
    html.Div([
        html.H4("Distribución de topicos y artistas"),
        dcc.Graph(id = 'artist-relevance',
                  figure = {'data': [gfx.Bar(
                        x = artist_relevance.index,
                        y = artist_relevance.values)]})
            ]),
    html.Div([
        html.H1("Principales palabras asociadas a los 10 tópicos"),
        dcc.Input(id='words-per-topic-input', type='number', placeholder=5),
        html.Table([html.Tr([html.Td(col) for col in words_per_topic.columns])]),
        html.Div(id='words-per-topic-container')
    ]),
    html.Div([
        html.H2("Distribución para un artista específico"),
        dcc.Dropdown(id='select-artist',
                     options = tidy_button_pairing(artist_relevance.index),
                     value='Pink Floyd'),
        dcc.Graph(id='visualize-artist')
    ]),
    html.Div([]),
    html.Div([
        html.H2("Distribución de tópicos para el artista"),
        dcc.Graph(id='topic-distribution-for-artist')
    ]),
    html.Div(),
    html.Div([
        html.H2("Distribución probabilística de tópicos"),
        dcc.Graph(id="probabilistic-topic-distribution-for-artist")
    ])
])


@app.callback(
    dash.dependencies.Output('words-per-topic-container', 'children'),
    [dash.dependencies.Input('words-per-topic-input', 'value')]
)
def generate_table(rows,dataframe=words_per_topic):
    body = [html.Tr([
        html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
    ]) for i in range(min(len(dataframe),rows))]
    return body


# # callback para words_per_topic
# @app.callback(
    # dash.dependencies.Output('words-per-topic-container-output', 'children'),
    # [dash.dependencies.Input('words-per-topics','data')])
# def update_graph(rows):
    # if rows is None:
        # dff = df
    # else:
        # dff = pd.DataFrame(rows)
    # return html.Div()



# Callback para ver un artista específico
@app.callback(
    dash.dependencies.Output('visualize-artist','figure'),
    [dash.dependencies.Input('select-artist', 'value')]
)
def update_graph(artist):
    tmp_df = labeled_data[labeled_data['artist'] == artist]
    tmp_df['song_length'] = tmp_df['lyrics'].apply(lambda x: len(x.split(' ')))
    return {
        'data': [gfx.Histogram(x=tmp_df['song_length'],
                               histnorm='percent',
                               name=f"Freq palabras para {artist}")],
        'layout': gfx.Layout(xaxis={'title': artist},
                             hovermode='closest')
    }

# topic distribution for song

@app.callback(
    dash.dependencies.Output('topic-distribution-for-artist', 'figure'),
    [dash.dependencies.Input('select-artist', 'value')]
)
def update_graph(artist):
    tmp_df = labeled_data[labeled_data['artist'] == artist]
    tmp_topics = tmp_df['highest_topic'].value_counts()
    return {
        'data': [gfx.Bar(x=tmp_topics.values,
                         y=[f"Tópico {i}" for i in tmp_topics.index],
                         name = f"Distribución de tópicos para {artist}",
                         orientation='h')],
        'layout': gfx.Layout(xaxis={'title': 'Cantidad de canciones pertenecientes a X tópico'},
                             title={'text': f"{artist}"}
                            )
    }

@app.callback(
    dash.dependencies.Output('probabilistic-topic-distribution-for-artist', 'figure'),
    [dash.dependencies.Input('select-artist', 'value')]
)
def update_graph(artist):
    tmp_df = labeled_data[labeled_data['artist'] == artist]
    tmp_topics = tmp_df.loc[:, list(filter(lambda x: 'Tópico' in x, tmp_df.columns))]
    list_of_topics = {}
    list_of_topics = {colname: gfx.Box(y=list(serie)) for colname, serie in tmp_topics.iteritems()}

    tmp_fig = gfx.Figure()
    for colname, serie in tmp_topics.iteritems():
        tmp_fig.add_trace(
            gfx.Box(x=serie,name=colname)
        )
    return tmp_fig


if __name__ == "__main__":
    app.run_server(debug=True)
