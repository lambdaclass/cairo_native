from argparse import ArgumentParser
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

argument_parser = ArgumentParser('Stress Test Plotter')
argument_parser.add_argument("logs_path")
arguments = argument_parser.parse_args()


def canonicalize(event):
    if event["fields"]["message"] != "finished round":
        return None

    return {
        "round": event["span"]["number"],
        "time": int(event["fields"]["time"]),
        "memory_used": event["fields"]["memory_used"],
        "cache_disk_size": event["fields"]["cache_disk_size"],
    }


def regression_line(label_x, label_y, data={}):
    b, a = np.polyfit(data[label_x], data[label_y], deg=1)
    xseq = np.linspace(min(data[label_x]), max(data[label_x]))

    return xseq, a + b * xseq


dataset = pd.read_json(arguments.logs_path, lines=True, typ="series")
dataset = dataset.map(canonicalize).dropna()
dataset = dataset.apply(pd.Series)

figure, axes = plt.subplots()

axes.scatter("round", "time", data=dataset)

x, y = regression_line("round", "time", dataset)
axes.plot(x, y, lw=2.5, color="k")

plt.show()
