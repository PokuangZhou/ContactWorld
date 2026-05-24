import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# =========================
# Change this part
# =========================

RESULTS = {
    "Wrist": {
        "Only": "rollout/wrist/rollout_metrics.json",
        "TacDepth": "rollout/wrist_tacdepth/rollout_metrics.json",
        "TacRGB": "rollout/wrist_tacdepth/rollout_metrics.json",
        "TacFF": "rollout/wrist_tacff/rollout_metrics.json",
    },
    "Front": {
        "Only": "rollout/front/rollout_metrics.json",
        "TacDepth": "rollout/front_tacdepth/rollout_metrics.json",
        "TacRGB": "rollout/front_tacrgb/rollout_metrics.json",
        "TacFF": "rollout/front_tacff/rollout_metrics.json",
    },
    "PointCloud": {
        "Only": "rollout/pc/rollout_metrics.json",
        "TacDepth": "rollout/pc_tacdepth/rollout_metrics.json",
        "TacRGB": "rollout/pc_tacrgb/rollout_metrics.json",
        "TacFF": "rollout/pc_tacff/rollout_metrics.json",
    },
}

OUTPUT_DIR = Path("rollout_comparison_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Style
# =========================

ACTION_COLORS = {
    "gt": "tab:blue",
    "random_uniform": "tab:orange",
    "zero": "tab:green",
}

TACTILE_LINESTYLES = {
    "Only": "-",
    "TacDepth": "--",
    "TacRGB": "-.",
    "TacFF": ":",
}

ACTION_ORDER = ["gt", "random_uniform", "zero"]
TACTILE_ORDER = ["Only", "TacDepth", "TacRGB", "TacFF"]


def load_metrics(path):
    path = Path(path)
    with open(path, "r") as f:
        return json.load(f)


def extract_steps_and_values(step_dict):
    steps = sorted(int(k.split("_")[0]) for k in step_dict.keys())
    values = [step_dict[f"{s}_step"] for s in steps]
    return steps, values


def get_global_ylim(results):
    all_values = []
    for vision_name, tactile_dict in results.items():
        for tactile_name, json_path in tactile_dict.items():
            metrics = load_metrics(json_path)
            for mode in metrics:
                all_values.extend(metrics[mode].values())

    ymin = 0.0
    ymax = max(all_values) * 1.08
    return ymin, ymax


def plot_one_vision(vision_name, tactile_jsons, ylim=None):
    plt.figure(figsize=(8, 5))

    for tactile_name in TACTILE_ORDER:
        if tactile_name not in tactile_jsons:
            continue

        metrics = load_metrics(tactile_jsons[tactile_name])

        for action_mode in ACTION_ORDER:
            if action_mode not in metrics:
                continue

            steps, values = extract_steps_and_values(metrics[action_mode])

            plt.plot(
                steps,
                values,
                color=ACTION_COLORS[action_mode],
                linestyle=TACTILE_LINESTYLES[tactile_name],
                marker="o",
                linewidth=2.0,
                markersize=4,
                alpha=0.95,
            )

    plt.title(f"{vision_name} Multi-step Rollout Error")
    plt.xlabel("Rollout step")
    plt.ylabel("Latent MSE")
    plt.grid(True, alpha=0.3)

    if ylim is not None:
        plt.ylim(*ylim)

    # Legend 1: action mode by color
    action_handles = [
        Line2D(
            [0], [0],
            color=ACTION_COLORS[m],
            linestyle="-",
            linewidth=2.5,
            label=m,
        )
        for m in ACTION_ORDER
    ]

    # Legend 2: tactile modality by linestyle
    tactile_handles = [
        Line2D(
            [0], [0],
            color="black",
            linestyle=TACTILE_LINESTYLES[t],
            linewidth=2.5,
            label=t,
        )
        for t in TACTILE_ORDER
    ]

    leg1 = plt.legend(
        handles=action_handles,
        title="Action mode",
        loc="upper left",
        frameon=True,
    )
    plt.gca().add_artist(leg1)

    plt.legend(
        handles=tactile_handles,
        title="Tactile modality",
        loc="lower right",
        frameon=True,
    )

    plt.tight_layout()

    png_path = OUTPUT_DIR / f"{vision_name.lower()}_rollout_12lines.png"
    pdf_path = OUTPUT_DIR / f"{vision_name.lower()}_rollout_12lines.pdf"

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def main():
    # Global y-limit across all vision/tactile/action combinations
    ylim = get_global_ylim(RESULTS)

    for vision_name, tactile_jsons in RESULTS.items():
        plot_one_vision(
            vision_name=vision_name,
            tactile_jsons=tactile_jsons,
            ylim=ylim,
        )


if __name__ == "__main__":
    main()