import os
import pickle

import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import plotly.express as px

style.use('classic')


def load_metrics(experiment_dir):
    metrics_path = os.path.join(experiment_dir, 'epoch_metrics.pkl')
    with open(metrics_path, 'rb') as file:
        data = pickle.load(file)
    return data


def plot_individual_metric(base_dir, metric_name, ylabel, ylim, file_suffix, bbox_to_anchor, ncol):
    plt.figure(figsize=(15, 4.8))
    color_palette = px.colors.qualitative.G10

    for i, experiment_name in enumerate(sorted(os.listdir(base_dir))):
        experiment_dir = os.path.join(base_dir, experiment_name)
        if os.path.isdir(experiment_dir):
            metrics = load_metrics(experiment_dir)
            epochs = np.arange(1, len(metrics['metrics']['train_loss']) + 1)

            if metric_name == 'train_acc':  # Special case for plotting train and test accuracy
                plt.plot(epochs, metrics['metrics']['train_acc'], label=f'{experiment_name} (Train)', color=color_palette[i % len(color_palette)])
                plt.plot(epochs, metrics['metrics']['test_acc'], label=f'{experiment_name} (Test)', linestyle='--', color=color_palette[i % len(color_palette)])
            else:
                plt.plot(epochs, metrics['metrics'][metric_name], label=f'{experiment_name}', color=color_palette[i % len(color_palette)])

    plt.title(ylabel)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.ylim(ylim)
    plt.legend(loc='upper center', bbox_to_anchor=bbox_to_anchor, ncol=ncol)
    pdf_path = os.path.join(base_dir, f'{metric_name}_{file_suffix}.png')
    plt.tight_layout()
    plt.savefig(pdf_path)
    # plt.show()
    plt.close()


def plot_all_metrics(base_dir='./Transformers/'):
    plot_individual_metric(base_dir, 'train_loss', 'Loss', [0, 0.12], 'loss_plot', bbox_to_anchor=(0.5, 1.25), ncol=6)
    plot_individual_metric(base_dir, 'train_acc', 'Accuracy', [0.94, 1.0], 'accuracy_plot', bbox_to_anchor=(0.5, 1.45), ncol=6)
    plot_individual_metric(base_dir, 'f1_score', 'F1 Score', [0.96, 0.985], 'f1_score_plot', bbox_to_anchor=(0.5, 1.25), ncol=6)


if __name__ == "__main__":
    plot_all_metrics()
