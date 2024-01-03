from typing import Dict, List, Any
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
from itertools import cycle

import pandas as pd
from argparse import ArgumentParser

import wandb
api = wandb.Api()


EMPTY_CELL_CHAR = "-"

# mi_max, mi_max_normalized, model_accuracy, model_accuracy_max
metric = "model_accuracy_max"
use_full = True
use_baseline = True

metric_labels = {
    "mi_max": "LBMI",
    "model_accuracy": "Accuracy",
    "model_accuracy_max": "LBA",
    "mi_max_normalized": "LBNMI",
}


def make_language_attribute_dict(embedding: str, cell: str, languages: List[str], empty_cell_char=EMPTY_CELL_CHAR):
    results_dict: Dict[str, Dict[str, Any]] = {}
    for l in languages:
        results_dict[l] = {}

        for a in attributes:
            result = data[data.attribute.eq(a) & data.language.eq(l) & data.embedding.eq(embedding)]

            assert len(result) <= 1

            if len(result) < 1:
                results_dict[l][a] = empty_cell_char
            else:
                # results_dict[l][a] = "{:.3f}".format(result[cell].iloc[0])
                results_dict[l][a] = result[cell].iloc[0]

    return results_dict


# Make bulk tables
# cell = "mi_max_normalized_50"
# e = "fasttext"
# results_dict: Dict[str, Dict[str, Any]] = make_language_attribute_dict(e, cell)

# Comparsion table
# cell = "mi_max_normalized_50"
# cell = "model_accuracy"
# results_dict = {}
# for l in languages:
#     results_dict[l] = {}

#     for a in attributes:
#         result = data[data.attribute.eq(a) & data.language.eq(l)]

#         assert len(result) <= 2

#         if len(result) < 1:
#             results_dict[l][a] = EMPTY_CELL_CHAR
#         else:
#             bert_res = result[data.embedding.eq("bert")][cell].iloc[0]
#             fasttext_res = result[data.embedding.eq("fasttext")][cell].iloc[0]
#             results_dict[l][a] = "{:.3f}".format(bert_res - fasttext_res)

# select_columns = ["Number", "Tense", "Gender and Noun Class", "Case", "Person", "Aspect"]
# df = pd.DataFrame(results_dict)
# print(df.transpose()[select_columns].to_latex())

# Log runs
# run = api.run("ltorroba/interp-bert/jedjua6o")
# if run.state == "finished":
#     for k in run.history(pandas=False):
#         if "model_accuracy" in k:
#             print(k["_step"], k["model_accuracy"])

def build_horizontal_bar_chart(group_languages: bool, languages: List[str], multicategory_group_names: List[str] = None):
    metric_names = []

    if use_baseline:
        metric_names.append("baseline_accuracy")

    dims = [2, 10, 50]
    metric_names.extend([f"{metric}_{i}" for i in dims])

    if use_full:
        metric_names.append("model_accuracy_full")

    group_names = languages if group_languages else attributes
    num_groups = len(group_names)
    vertical_spacing = 0.3 / num_groups
    fig = make_subplots(
        rows=num_groups, cols=1, shared_xaxes=True, shared_yaxes=False,
        vertical_spacing=vertical_spacing)

    # Build dataframe
    embedding_dfs = {}

    for e in embeddings:
        agg_metric_dfs = {}

        for metric_name in metric_names:
            agg_metric_df = pd.DataFrame(make_language_attribute_dict(e, metric_name, languages, np.NaN))
            if not group_languages:
                agg_metric_df = agg_metric_df.transpose()

            agg_metric_dfs[metric_name] = agg_metric_df.mean()

        metric_df = pd.concat(agg_metric_dfs, axis=1)
        embedding_dfs[e] = metric_df

    df = pd.concat(embedding_dfs).swaplevel(0, 1)

    # Generate color array
    colors = []
    darken = 1.0
    darkening_factor = 0.75
    for idx, _ in enumerate(metric_names):
        dim_colors = []
        from utils.graph_writer import QUALITATIVE_COLORS
        for e, color in zip(embeddings, cycle(QUALITATIVE_COLORS)):
            rgb_ints = [str(int(x.strip(" ")) * darken) for x in color[4:][:-1].split(",")]
            new_color = f"rgb({','.join(rgb_ints)})"
            dim_colors.append(new_color)

        colors.append(dim_colors)

        darken *= darkening_factor

    # Build data array
    annotations = []
    # df_groups = {group: group_df for group, group_df in df.groupby(level=0)}
    for idx, group in enumerate(group_names):
        for metric_idx, metric_name in enumerate(metric_names):

            style_data = {
                "marker_color": colors[metric_idx],
                # "name": f"{e}-{d}",
                "orientation": "h",
                "showlegend": False,
            }

            # Picture relative difference
            if metric_idx == 0:
                vals = df.loc[group][f"{metric_name}"]
            else:
                vals = (df.loc[group][f"{metric_name}"] - df.loc[group][f"{metric_names[metric_idx - 1]}"]).clip(0, 1)

            # graph_data.append(go.Bar(
            fig.add_trace(go.Bar(
                y=df.loc[group].index.values.tolist(),
                x=vals,
                **style_data),
                row=idx + 1, col=1
            )

        fig.update_yaxes(row=idx + 1, col=1, showticklabels=False, range=[0.0, 1.0])

        group_label = group if group_languages else attributes_labels[group]
        annotations.append(
            dict(
                x=-0.01, y=0.5,
                xref='paper',
                yref=f'y{idx + 1}',
                text=group_label,
                showarrow=False,
                xanchor="right",
                yanchor="middle"
            )
        )

    if multicategory_group_names:
        # Build array: (group_name, start_id, end_id)
        group_indices = {}
        for idx, group_name in enumerate(multicategory_group_names[::-1]):
            if group_name not in group_indices:
                group_indices[group_name] = []

            group_indices[group_name].append(idx)

        group_ranges = []
        for group_name, idcs in group_indices.items():
            group_ranges.append((group_name, min(idcs), max(idcs) + 1))

        # Add lines and labels for each group
        for group_name, start_id, end_id in group_ranges:
            start_item_num = start_id
            end_item_num = end_id
            bar_height = (1.0 - (num_groups - 1) * vertical_spacing) / num_groups

            # line_x = -0.035
            line_x = -0.10
            start_y = start_item_num * (bar_height + vertical_spacing)
            end_y = end_item_num * (bar_height + vertical_spacing) - vertical_spacing
            fig.add_shape(
                type="line",
                xref="paper",
                yref="paper",
                x0=line_x,
                y0=start_y,
                x1=line_x,
                y1=end_y,
                line=dict(
                    color="Black",
                    width=1,
                ),
            )

            annotations.append(
                dict(
                    x=line_x - 0.01, y=(start_y + end_y) / 2,
                    xref='paper',
                    yref='paper',
                    text=f"{group_name}",
                    showarrow=False,
                    xanchor="right",
                    yanchor="middle"
                )
            )

    fig.update_xaxes(row=idx + 1, col=1, range=[0.0, 1.0], title=metric_labels[metric])

    if group_languages:
        if group_names:
            left_margin = 120
        else:
            left_margin = 50
    else:
        left_margin = 70

    fig.update_layout(
        margin=dict(l=left_margin, r=0, t=0, b=0),
        annotations=annotations,
        barmode='stack',
        bargap=0.0,
        font=dict(family="serif")
    )

    return fig

iso_to_wals = {
    'afr': 'afr',
    'eus': 'bsq',
    'bel': 'blr',
    'bul': 'bul',
    'cat': 'ctl',
    'ces': 'cze',
    'dan': 'dsh',
    'nld': 'dut',
    'eng': 'eng',
    'est': 'est',
    'fin': 'fin',
    'fra': 'fre',
    'deu': 'ger',
    'heb': 'heb',
    'hin': 'hin',
    'hun': 'hun',
    'gle': 'iri',
    'ita': 'ita',
    'lav': 'lat',
    'lit': 'lit',
    'ron': 'mol',
    'pol': 'pol',
    'por': 'por',
    'ara': 'ams',
    # 'tam': 'tml',
    'slk': 'svk',
    'lat': 'spa',
    'fas': 'prs',
    'slv': 'slo',
    'hrv': 'scr',
    'spa': 'spa',
    'swe': 'swe',
    'rus': 'rus',
    'srp': 'scr',
    'tur': 'tur',
    'ukr': 'ukr',
    'urd': 'urd',
}

families = {
    'afr': 'Indo-European',
    'ams': 'Afro-Asiatic',
    'bsq': 'Basque',
    'blr': 'Indo-European',
    'bul': 'Indo-European',
    'ctl': 'Indo-European',
    'cze': 'Indo-European',
    'dsh': 'Indo-European',
    'dut': 'Indo-European',
    'eng': 'Indo-European',
    'est': 'Uralic',
    'fin': 'Uralic',
    'fre': 'Indo-European',
    'ger': 'Indo-European',
    'heb': 'Afro-Asiatic',
    'hin': 'Indo-European',
    'hun': 'Uralic',
    'iri': 'Indo-European',
    'ita': 'Indo-European',
    'lat': 'Indo-European',
    'lit': 'Indo-European',
    'mol': 'Indo-European',
    'prs': 'Indo-European',
    'pol': 'Indo-European',
    'por': 'Indo-European',
    'rus': 'Indo-European',
    'scr': 'Indo-European',
    'slo': 'Indo-European',
    'svk': 'Indo-European',
    'spa': 'Indo-European',
    'swe': 'Indo-European',
    'tml': 'Dravidian',
    'tur': 'Altaic',
    'ukr': 'Indo-European',
    'urd': 'Indo-European'
}

genus = {
    'afr': 'Germanic',
    'ams': 'Semitic',
    'bsq': 'Basque',
    'blr': 'Slavic',
    'bul': 'Slavic',
    'ctl': 'Romance',
    'cze': 'Slavic',
    'dsh': 'Germanic',
    'dut': 'Germanic',
    'eng': 'Germanic',
    'est': 'Finnic',
    'fin': 'Finnic',
    'fre': 'Romance',
    'ger': 'Germanic',
    'heb': 'Semitic',
    'hin': 'Indic',
    'hun': 'Ugric',
    'iri': 'Celtic',
    'ita': 'Romance',
    'lat': 'Baltic',
    'lit': 'Baltic',
    'mol': 'Romance',
    'prs': 'Iranian',
    'pol': 'Slavic',
    'por': 'Romance',
    'rus': 'Slavic',
    'scr': 'Slavic',
    'svk': 'Slavic',
    'slo': 'Slavic',
    'spa': 'Romance',
    'swe': 'Germanic',
    'tml': 'Southern Dravidian',
    'tur': 'Turkic',
    'ukr': 'Slavic',
    'urd': 'Indic'
}

merged = {}
for k, v in families.items():
    if v == "Indo-European":
        merged[k] = f"IE ({genus[k]})"
    else:
        merged[k] = v

lang_sorted_family = sorted([(code, merged[iso_to_wals[code]]) for code in iso_to_wals], key=lambda x: (x[1], x[0]))
print(lang_sorted_family)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("file")
    args = parser.parse_args()

    data = pd.read_csv(args.file)
    # print(data.head())

    # attributes = sorted(list(set(data["attribute"])))
    attributes = sorted(["Person", "Animacy", "Aspect", "Tense", "Case", "Number", "Gender and Noun Class", "Voice", "Definiteness"])
    _languages = sorted(list(set(data["language"])))
    # embeddings = sorted(list(set(data["embedding"])))
    embeddings = ["bert", "fasttext"]

    attributes_labels = {a: a for a in attributes}
    attributes_labels["Gender and Noun Class"] = "Gender"

    # Make bar chart (attributes)
    fig_attributes = build_horizontal_bar_chart(group_languages=False, languages=_languages)
    fig_attributes.write_image(f"images/fig_attributes.{metric}.pdf", width=400, height=200)
    # fig_attributes.show()

    fig_langs = build_horizontal_bar_chart(group_languages=True, languages=_languages)
    fig_langs.write_image(f"images/fig_languages.{metric}.pdf", width=400, height=800)
    # fig_langs.show()

    # feature = "family"
    # wals_df = Wals('83a').get_df()
    # wals_df = wals_df[wals_df["wals_code"].isin([iso_to_wals[l] for l in _languages])]
    # print(wals_df)

    # Make languages plot grouped by lang family
    languages = [x[0] for x in lang_sorted_family]
    group_names = [x[1] for x in lang_sorted_family]
    print(languages)
    fig_langs = build_horizontal_bar_chart(group_languages=True, languages=languages, multicategory_group_names=group_names)
    fig_langs.write_image(f"images/fig_languages_group_family.{metric}.pdf", width=400, height=600)
    # fig_langs.show()


