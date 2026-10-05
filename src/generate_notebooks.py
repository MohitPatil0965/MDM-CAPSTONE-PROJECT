"""Utility script to generate standard thin Jupyter notebooks."""

import json
from pathlib import Path


def generate_notebooks():
    notebooks = {
        "01_eda.ipynb": {
            "title": "01 - Exploratory Data Analysis & Preprocessing",
            "code": (
                "# 01 Exploratory Data Analysis & Cleaning\n"
                "import pandas as pd\n"
                "from src.data_prep import prepare_data\n\n"
                "train_df, test_df, full_df = prepare_data()\n"
                "print('Cleaned dataset shape:', full_df.shape)\n"
                "print('Response rate:', full_df['Response'].mean())\n"
                "full_df.head()\n"
            ),
        },
        "02_segmentation.ipynb": {
            "title": "02 - Customer Segmentation (K-Means)",
            "code": (
                "# 02 Audience Segmentation\n"
                "import pandas as pd\n"
                "from src.data_prep import prepare_data\n"
                "from src.segmentation import perform_segmentation\n\n"
                "train_df, test_df, full_df = prepare_data()\n"
                "tr_seg, te_seg, fu_seg, profiles_df, artifacts = perform_segmentation(train_df, test_df, full_df)\n"
                "print('Optimal K:', artifacts['best_k'])\n"
                "profiles_df\n"
            ),
        },
        "03_modeling.ipynb": {
            "title": "03 - Predictive Modeling & Calibration",
            "code": (
                "# 03 LightGBM Modeling & Calibration\n"
                "import pandas as pd\n"
                "from src.data_prep import prepare_data\n"
                "from src.segmentation import perform_segmentation\n"
                "from src.model import train_pipeline\n\n"
                "tr, te, fu = prepare_data()\n"
                "tr_seg, te_seg, fu_seg, _, _ = perform_segmentation(tr, te, fu)\n"
                "model, scored_df, metrics = train_pipeline(tr_seg, te_seg, fu_seg)\n"
                "for k, v in metrics.items():\n"
                "    print(f'{k}: {v}')\n"
            ),
        },
        "04_optimization.ipynb": {
            "title": "04 - Multi-Objective Optimization (NSGA-II)",
            "code": (
                "# 04 NSGA-II Multi-Objective Optimization\n"
                "import pandas as pd\n"
                "from src.optimizer import run_nsga2_optimization, select_knee_point, select_topsis_solution\n\n"
                "scored_df = pd.read_parquet('data/processed/scored_campaign_data.parquet')\n"
                "pareto_df, alloc_df, stats, helper = run_nsga2_optimization(scored_df)\n"
                "print(f'Generated {len(pareto_df)} Pareto optimal solutions.')\n"
                "knee_idx = select_knee_point(pareto_df)\n"
                "print('Recommended Knee Solution ID:', knee_idx)\n"
                "pareto_df.head()\n"
            ),
        },
        "05_results.ipynb": {
            "title": "05 - Benchmark Evaluation & Results",
            "code": (
                "# 05 Strategy Comparison & Sensitivity\n"
                "import pandas as pd\n"
                "from src.evaluate import run_full_evaluation\n\n"
                "run_full_evaluation()\n"
                "master_df = pd.read_csv('reports/tables/master_comparison.csv')\n"
                "master_df\n"
            ),
        },
    }

    Path("notebooks").mkdir(exist_ok=True)
    for fname, content in notebooks.items():
        nb = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [f"# {content['title']}\n"],
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [line + "\n" for line in content["code"].split("\n")],
                },
            ],
            "metadata": {
                "language_info": {"name": "python"}
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        with open(Path("notebooks") / fname, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=2)

    print("Successfully created 5 Jupyter notebooks in notebooks/ directory.")


if __name__ == "__main__":
    generate_notebooks()
