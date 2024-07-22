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
        "memory used": int(event["fields"]["memory_used"]) / 2**20,
        "cache disk size": int(event["fields"]["cache_disk_size"]) / 2**20,
    }


dataset = dataset.map(canonicalize).dropna().apply(pd.Series)


def trend_line(label_x, label_y, data={}, degrees=1):
    coefficients = np.polyfit(data[label_x], data[label_y], deg=degrees)[::-1]
    xseq = np.linspace(min(data[label_x]), max(data[label_x]))

    yseq = sum([coefficient * xseq ** degree for degree,
               coefficient in enumerate(coefficients)])

    return xseq, yseq


figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
figure.tight_layout(pad=2)

axes[0].scatter("round", "time", data=dataset, s=5, alpha=0.1)
x, y = trend_line("round", "time", dataset, degrees=3)
axes[0].plot(x, y, lw=2.5, color="k")
axes[0].set_xlabel('Round')
axes[0].set_ylabel('Time [ms]')
axes[0].set_title('Compilation Time')


axes[1].plot("round", "memory used", data=dataset)
axes[1].plot("round", "cache disk size", data=dataset)
axes[1].set_xlabel("Round")
axes[1].set_ylabel("Megabytes")
axes[1].set_title('Space Usage')
axes[1].legend()

plt.show()
