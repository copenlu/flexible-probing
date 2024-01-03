from typing import Dict, List, Tuple
import plotly.graph_objects as go
from plotly.colors import qualitative
import json
from itertools import cycle
from argparse import ArgumentParser
from os import listdir
from os.path import isfile, join
import re
import numpy as np
import random


QUALITATIVE_COLORS = qualitative.Set2
QUALITATIVE_COLORS2 = qualitative.Dark2
LANGUAGE_LABELS = {
    "eng": "English",
    "por": "Portuguese",
    "spa": "Spanish",
    "pol": "Polish",
    "deu": "German",
    "rus": "Russian",
    "heb": "Hebrew",
    "ara": "Arabic",
    "tur": "Turkish",
    "fin": "Finnish",
}

TRAINER_LABELS = {
    "lowerbound": "Dalvi",
    "upperbound": "Upperbound",
    "conditional-poisson": "Conditional Poisson",
    "qda": "Gaussian",
    "poisson": "Poisson",
}

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "lb_accuracy": "LBA",
    "mi": "NMI",
    "lb_mi": "LBNMI",
}

TRAINER_COLORS = {
    "lowerbound": QUALITATIVE_COLORS[3],
    "upperbound": QUALITATIVE_COLORS[4],
    "conditional-poisson": QUALITATIVE_COLORS[2],
    "poisson": QUALITATIVE_COLORS[1],
    "qda": QUALITATIVE_COLORS[0],
}

DEFAULT_RESULTS_FOLDER = "experiments/02"
DEFAULT_FILE_FORMAT = "experiments/02/02_{lang}-{attribute}-{trainer}-results.json"


parser = ArgumentParser(description="Generates line graph of average MI.")
parser.add_argument("metric", choices=list(METRIC_LABELS.keys()), help="The metric to be plotted.")
parser.add_argument("--trainers", nargs="+", choices=list(TRAINER_LABELS.keys()), required=True,
                    help="The trainers to be plotted (multiple values possible). The legend will show them \
                    in the order provided here.")
parser.add_argument("--input-file-name-format", default=DEFAULT_FILE_FORMAT,
                    help="If you want to read from custom-made files, you can change this variable. \
                    Instances of {lang}, {attribute}, {trainer} are replaced with the language, \
                    attribute and trainer, in the format expected by this command.")
parser.add_argument("--no-show", default=False, action="store_true", help="Pass this flag to disable opening \
                    of the graph on the window.")
parser.add_argument("--num-example-lines", default=10, type=int, help="How many real examples do we want to \
                    display (that compose the average).")
parser.add_argument("--y-min", default=0.0, type=float, help="Minimum Y.")
parser.add_argument("--y-max", default=1.0, type=float, help="Minimum Y.")
parser.add_argument("--output-pdf", type=str, help="Output filename for the PDF image.")
args = parser.parse_args()


# Build list of results that show be processed
file_list = [f for f in listdir(DEFAULT_RESULTS_FOLDER)
             if isfile(join(DEFAULT_RESULTS_FOLDER, f)) and ".pdf" not in f]
RESULTS = []
for f in file_list:
    match = re.search(r"^02_([a-zA-Z]*)-([a-zA-Z ]*)-([\-a-zA-Z]*)-results.json$", f)
    lang = match.group(1)
    attribute = match.group(2)
    trainer = match.group(3)

    RESULTS.append((lang, attribute, trainer))

RESULTS = [(l, a, t) for (l, a, t) in RESULTS if t in args.trainers]

# Setup local variables
metric, metric_label = args.metric, METRIC_LABELS[args.metric]
trainers_labelled = [(t, TRAINER_LABELS[t]) for t in args.trainers]
file_format = args.input_file_name_format

# Load experiment results for the specified trainers and group the relevant metric
trainer_metrics: Dict[Tuple[str, str, str], Dict[str, List[float]]] = {}
for lang, attribute, trainer in RESULTS:
    with open(file_format.format(lang=lang, attribute=attribute, trainer=trainer), "r") as h:
        file_data = json.load(h)

    trainer_metrics[(lang, attribute, trainer)] = {
        "mi": [],
        "accuracy": [],
        "lb_mi": [],
        "lb_accuracy": [],
    }

    lb_mi = file_data[0]["metrics"]["mi"]
    lb_acc = file_data[0]["metrics"]["accuracy"]
    for step in file_data:
        for k, v in step["metrics"].items():
            trainer_metrics[(lang, attribute, trainer)][k].append(v)

        if step["metrics"]["mi"] >= lb_mi:
            lb_mi = step["metrics"]["mi"]

        if step["metrics"]["accuracy"] >= lb_acc:
            lb_acc = step["metrics"]["accuracy"]

        trainer_metrics[(lang, attribute, trainer)]["lb_mi"].append(lb_mi)
        trainer_metrics[(lang, attribute, trainer)]["lb_accuracy"].append(lb_acc)


# Compute average of all
average_metrics = {}
for tr in args.trainers:
    average_metrics[tr] = {}
    for metric in ["mi", "accuracy", "lb_mi", "lb_accuracy"]:
        average_metrics[tr][metric] = np.array(
            [v[metric] for (_, _, t), v in trainer_metrics.items() if t == tr])
        average_metrics[tr][metric] = np.mean(average_metrics[tr][metric], axis=0)


layout = go.Layout(
    font=dict(family="serif", color="black"),
    margin=dict(l=0, r=0, t=0, b=0),
    xaxis_title="Number of sampled dimensions",
    yaxis=dict(title=metric_label, range=[args.y_min, args.y_max])
)
fig = go.Figure(layout=layout)
x = list(range(1, 101))

# Add example lines
num_example_lines = args.num_example_lines
if "lb_" in args.metric:
    metric_unbound = args.metric[3:]
    random.seed(2)
    lang_attr = sorted(list(set([(lang, attribute) for (lang, attribute, _) in trainer_metrics.keys()])))
    random.shuffle(lang_attr)
    lang_attr = lang_attr[:num_example_lines]

    for trainer, trainer_label in trainers_labelled:
        trainer_color = TRAINER_COLORS[trainer]
        for lang, attr in lang_attr:
            y = trainer_metrics[(lang, attr, trainer)][metric_unbound]
            fig.add_trace(go.Scatter(x=x, y=y, marker_color=trainer_color, mode="lines", opacity=0.6,
                                     line_width=1))


# Dimension grouped metrics
for trainer, trainer_label in trainers_labelled:
    trainer_color = TRAINER_COLORS[trainer]
    y = average_metrics[trainer][args.metric]
    fig.add_trace(go.Scatter(x=x, y=y, name=trainer_label, marker_color=trainer_color, mode="lines",
                             line_width=3))


if not args.no_show:
    fig.show()

if args.output_pdf:
    fig.update_layout({
        "showlegend": False,
        "width": 350,
        "height": 200,
    })
    fig.write_image(args.output_pdf)
