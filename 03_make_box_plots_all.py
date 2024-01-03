from typing import Dict, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import qualitative
import json
from itertools import cycle
from argparse import ArgumentParser


QUALITATIVE_COLORS = qualitative.Set2
VIVID_COLORS = qualitative.Alphabet
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

ACTIVATION_LABELS = {
    "sigmoid1": "Linear",
    "relu2": "MLP-1",
    "relu3": "MLP-2"
}

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "mi": "Normalized Mutual Information",
}

ACTIVATION_COLORS = {
    "MLP-1": QUALITATIVE_COLORS[7],
    "MLP-2": QUALITATIVE_COLORS[5],
    "Linear": QUALITATIVE_COLORS[2],
}

DEFAULT_FILE_FORMAT = "experiments/03/03_{lang}-{attribute}-{activation}-results.json"

RESULTS = [("eng", "Number"), ("por", "Gender and Noun Class"), ("pol", "Tense"), ("rus", "Voice"),
           ("ara", "Case"), ("fin", "Case")]


parser = ArgumentParser(description="Generates a box plot to compare the robustness of different activation functions. \
                        The graph is generated for one language, and the results are plotted for different \
                        numbers of dimensions (shown on the X-axis) and different activations (shown in \
                        different colors), for a specific metric (e.g., accuracy) and UniMorph attribute, \
                        (e.g., \"Number\" or \"Gender and Noun Class\"). This corresponds to Figure 1 in the \
                        paper.")
parser.add_argument("metric", choices=list(METRIC_LABELS.keys()), help="The metric to be plotted.")
parser.add_argument("--activations", nargs="+", choices=list(ACTIVATION_LABELS.keys()), required=True,
                    help="The activations to be plotted (multiple values possible). The legend will show them \
                    in the order provided here.")
parser.add_argument("--input-file-name-format", default=DEFAULT_FILE_FORMAT,
                    help="If you want to read from custom-made files, you can change this variable. \
                    Instances of {lang}, {attribute}, {activation} are replaced with the language, \
                    attribute and activation, in the format expected by this command.")
parser.add_argument("--no-show", default=False, action="store_true", help="Pass this flag to disable opening \
                    of the graph on the window.")
parser.add_argument("--output-pdf", type=str, help="Output filename for the PDF image.")
args = parser.parse_args()


# Validate layout
rows = 2
cols = 3
assert rows * cols == len(RESULTS)


metric, metric_label = args.metric, METRIC_LABELS[args.metric]
fig = make_subplots(rows=rows, cols=cols, shared_yaxes=True,
                    subplot_titles=[f"{attr} ({lang})" for lang, attr in RESULTS],
                    # specs = [[{} for _ in range(cols)] for _ in range(rows)],
                    vertical_spacing=0.10, 
                    # horizontal_spacing=0.05
                    x_title="Number of sampled dimensions", y_title=metric_label)

for idx, (language, attribute) in enumerate(RESULTS):
    cols_id = idx % cols + 1
    rows_id = idx // cols + 1

    print(f"({rows_id}, {cols_id}): {language}--{attribute}")

    # Setup local variables
    lang, lang_label = language, LANGUAGE_LABELS[language]
    activations_labelled = [(t, ACTIVATION_LABELS[t]) for t in args.activations]
    file_format = args.input_file_name_format
    attribute, attribute_label = attribute, attribute

    # Load experiment results for the specified activations and group the relevant metric
    activation_grouped_metrics: Dict[str, Dict[int, float]] = {}
    for activation, activation_label in activations_labelled:
        with open(file_format.format(lang=lang, attribute=attribute, activation=activation), "r") as h:
            file_data = json.load(h)

        # Save each experiments result for the tracked metric
        activation_metrics: Dict[int, List[float]] = {}
        for experiment in file_data:
            if (dimensions_sampled := len(experiment["dimensions"])) not in activation_metrics:  # noqa
                activation_metrics[dimensions_sampled] = []

            activation_metrics[dimensions_sampled].append(experiment["metrics"][metric])

        activation_grouped_metrics[activation] = activation_metrics

    # Activation grouped metrics
    for activation, activation_label in activations_labelled:
        print(activation_label)
        x = []
        y = []
        for dims, results in activation_grouped_metrics[activation].items():
            x += [str(dims)] * len(results)
            y += results

        fig.append_trace(go.Box(
            x = x,
            y = y,
            name = activation_label,
            marker_color = ACTIVATION_COLORS[activation_label]
        ), rows_id, cols_id)

    fig.update_yaxes(range=[0.0, 1.0], row=rows_id, col=cols_id)
    fig.update_xaxes(type="category", row=rows_id, col=cols_id)


layout = go.Layout(
    margin=dict(t=30, b=60, l=60, r=0),
    font=dict(family="serif", color="black"),
    showlegend=False,
)
fig.update_layout(layout)

if not args.no_show:
    fig.show()

fig.update_layout({
        "showlegend": False,
        "height": 600,
})
fig.write_image("experiments/03/03_boxplots_all.pdf")
