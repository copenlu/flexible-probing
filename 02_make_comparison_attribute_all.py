from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import json
from itertools import cycle
from argparse import ArgumentParser
import plotly.express as px
import plotly.graph_objects as go

from os import listdir
from os.path import isfile, join
import re
import functools


LANGUAGE_LABELS = {
    "eng": "English",
    "por": "Portuguese",
    "pol": "Polish",
    "ara": "Arabic",
    "rus": "Russian",
    "fin": "Finnish",
}


METRIC_LABELS = {
    "accuracy": "Accuracy",
    "mi": "Normalized Mutual Information",
}

attributes = ['Animacy', 'Aspect', 'Case', 'Definiteness', 'Gender and Noun Class', 'Mood', 'Number', 'Person', 'Tense', 'Voice']

DEFAULT_RESULTS_FOLDER =  "experiments/02/"
DEFAULT_FILE_FORMAT = "experiments/02/02_{lang}-{attribute}-conditional-poisson-results.json"

parser = ArgumentParser(description="Generates a box plot to compare the robustness of different trainers. \
                        The graph is generated for one language, and the results are plotted for different \
                        numbers of dimensions (shown on the X-axis) and different trainers (shown in \
                        different colors), for a specific metric (e.g., accuracy) and UniMorph attribute, \
                        (e.g., \"Number\" or \"Gender and Noun Class\"). This corresponds to Figure 1 in the \
                        paper.")
parser.add_argument("metric", choices=list(METRIC_LABELS.keys()), help="The metric to be plotted.")
parser.add_argument("attribute", type=str, help="The attribute to be plotted. This should be some UniMorph \
                    dimension.")
parser.add_argument("--input-file-name-format", default=DEFAULT_FILE_FORMAT,
                    help="If you want to read from custom-made files, you can change this variable. \
                    Instances of {lang}, {attribute}, {trainer} are replaced with the language, \
                    attribute and trainer, in the format expected by this command.")
parser.add_argument("--no-show", default=False, action="store_true", help="Pass this flag to disable opening \
                    of the graph on the window.")
parser.add_argument("--y-min", default=0.0, type=float, help="Minimum Y.")
parser.add_argument("--y-max", default=1.0, type=float, help="Minimum Y.")
parser.add_argument("--output-pdf", type=str, help="Output filename for the PDF image.")
args = parser.parse_args()


file_list = [f for f in listdir(DEFAULT_RESULTS_FOLDER)
             if isfile(join(DEFAULT_RESULTS_FOLDER, f)) and ".json" in f]
RESULTS = []
file_format = args.input_file_name_format


data = []
for lang in LANGUAGE_LABELS.keys():
    try:
        with open(file_format.format(lang=lang, attribute=args.attribute), "r") as h:
                file_data = json.load(h)
                for experiment in file_data:
                    if experiment['dimensions'] != "ALL":
                        data.append({
                            'language': lang,
                            'attribute': args.attribute,
                            'no_dim': len(experiment['dimensions']),
                            'metric': experiment["metrics"][args.metric],
                        })
    except:
        print(lang, ' -- ', args.attribute, ' pair not available')

data = pd.DataFrame(data).groupby(['no_dim', 'language']).mean().reset_index()


layout = go.Layout(
    font=dict(family="serif", color="black"),
    margin=dict(l=0, r=0, t=0, b=0),
    xaxis_title="Number of sampled dimensions",
    yaxis=dict(title=METRIC_LABELS[args.metric], range=[args.y_min, args.y_max])
)
fig = go.Figure(layout=layout)
x = list(range(1, 101))

for lang in data.language.unique():
    data_lang = data[data.language == lang]
    fig.add_trace(go.Scatter(x=data_lang.no_dim, y=data_lang.metric, name = lang, mode='lines', line_width=3))

if not args.no_show:
    fig.show()

if args.output_pdf:
    fig.update_layout({
        "showlegend": True,
        "width": 900,
        "height": 500,
        "font": dict(
            size=26,
        ),
        "xaxis": dict(
            tickvals=[10, 50, 100, 250, 500],
        )
    })
    fig.update_layout(legend=dict(
        orientation = "h",
        y=1,
    ))
    fig.write_image(args.output_pdf)
