from argparse import ArgumentParser
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

argument_parser = ArgumentParser('Stress Test Plotter')
argument_parser.add_argument("logs_path")
arguments = argument_parser.parse_args()


dataset = pd.read_json(arguments.logs_path, lines=True, typ="series")


def canonicalize(event):
    if event["fields"]["message"] != "finished round":
        return None

    return {
        "round": int(event["span"]["number"]),
        "time": int(event["fields"]["time"]),
        "memory_used": int(event["fields"]["memory_used"]),
        "cache_disk_size": int(event["fields"]["cache_disk_size"]),
    }


dataset = dataset.map(canonicalize).dropna().apply(pd.Series)


def trend_line(label_x, label_y, data={}, degrees=1):
    coefficients = np.polyfit(data[label_x], data[label_y], deg=degrees)[::-1]
    xseq = np.linspace(min(data[label_x]), max(data[label_x]))

    yseq = sum([coefficient * xseq ** degree for degree,
               coefficient in enumerate(coefficients)])

    return xseq, yseq


figure, axes = plt.subplots()

axes.scatter("round", "time", data=dataset, s=5)

x, y = trend_line("round", "time", dataset, degrees=3)
axes.plot(x, y, lw=2.5, color="k")

plt.show()
