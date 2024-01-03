from typing import Dict, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import qualitative
import json
from itertools import cycle
from argparse import ArgumentParser


QUALITATIVE_COLORS = qualitative.Set2
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
    "poisson": "Poisson",
    "sigmoid1": "Conditional Poisson",
    "relu2": "MLP1",
    "relu3": "MLP2"
}

TRAINER_COLORS = {
    "Dalvi": QUALITATIVE_COLORS[3],
    "Upperbound": QUALITATIVE_COLORS[4],
    "Conditional Poisson": QUALITATIVE_COLORS[2],
    "Poisson": QUALITATIVE_COLORS[1],
    "Adaptive": QUALITATIVE_COLORS[5],
    "Gaussian": QUALITATIVE_COLORS[0],
    "MLP1": QUALITATIVE_COLORS[5],
    "MLP2": QUALITATIVE_COLORS[7]
}

TRAINER_COLORS = {
    "Dalvi": QUALITATIVE_COLORS[3],
    "Upperbound": QUALITATIVE_COLORS[4],
    "Fixed": QUALITATIVE_COLORS[0],
    "Flexible": QUALITATIVE_COLORS[1],
    "Adaptive": QUALITATIVE_COLORS[5],
    "Gaussian": QUALITATIVE_COLORS[2],
}


METRIC_LABELS = {
    "accuracy": "Accuracy",
    "mi": "Normalized Mutual Information",
}

DEFAULT_FILE_FORMAT = "experiments/01/01_{lang}-{attribute}-{trainer}-results.json"

RESULTS = [("eng", "Number"), ("por", "Gender and Noun Class"), ("pol", "Tense"), ("rus", "Voice"),
           ("ara", "Case"), ("fin", "Case")]


parser = ArgumentParser(description="Generates a box plot to compare the robustness of different trainers. \
                        The graph is generated for one language, and the results are plotted for different \
                        numbers of dimensions (shown on the X-axis) and different trainers (shown in \
                        different colors), for a specific metric (e.g., accuracy) and UniMorph attribute, \
                        (e.g., \"Number\" or \"Gender and Noun Class\"). This corresponds to Figure 1 in the \
                        paper.")
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
                    vertical_spacing=0.2, horizontal_spacing=0.05,
                    x_title="Number of sampled dimensions",
                    y_title=metric_label)

for idx, (language, attribute) in enumerate(RESULTS):
    cols_id = idx % cols + 1
    rows_id = idx // cols + 1

    print(f"({rows_id}, {cols_id}): {language}--{attribute}")

    # Setup local variables
    lang, lang_label = language, LANGUAGE_LABELS[language]
    trainers_labelled = [(t, TRAINER_LABELS[t]) for t in args.trainers]
    file_format = args.input_file_name_format
    attribute, attribute_label = attribute, attribute

    # Load experiment results for the specified trainers and group the relevant metric
    trainer_grouped_metrics: Dict[str, Dict[int, float]] = {}
    for trainer, trainer_label in trainers_labelled:
        with open(file_format.format(lang=lang, attribute=attribute, trainer=trainer), "r") as h:
            file_data = json.load(h)

        # Save each experiments result for the tracked metric
        trainer_metrics: Dict[int, List[float]] = {}
        for experiment in file_data:
            if (dimensions_sampled := len(experiment["dimensions"])) not in trainer_metrics:  # noqa
                trainer_metrics[dimensions_sampled] = []

            trainer_metrics[dimensions_sampled].append(experiment["metrics"][metric])

        trainer_grouped_metrics[trainer] = trainer_metrics

    # Trainer grouped metrics
    for (trainer, trainer_label) in trainers_labelled:
        x = []
        y = []
        for dims, results in trainer_grouped_metrics[trainer].items():
            x += [str(dims)] * len(results)
            y += results

        fig.append_trace(go.Box(
            x = x,
            y = y,
            name = trainer_label,
            marker_color = TRAINER_COLORS[trainer_label]
        ), rows_id, cols_id)

    fig.update_yaxes(range=[0.0, 1.0], row=rows_id, col=cols_id)
    fig.update_xaxes(type="category", row=rows_id, col=cols_id)


layout = go.Layout(
    # margin=dict(t=20, b=0, l=60, r=0),  # if we remove x-axis label
    margin=dict(t=20, b=50, l=60, r=0),  # if we add the x-axis label
    font=dict(family="serif", color="black"),
    showlegend=False,
)
fig.update_layout(layout)

if not args.no_show:
    fig.show()

if args.output_pdf:
    fig.update_layout({
        "showlegend": False,
        "height": 300,
    })
    fig.write_image(args.output_pdf)
