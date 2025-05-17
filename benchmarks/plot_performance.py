#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import sys

def load_sections(csv_file):
    """
    Parses a CSV with sections of the form:
      FunctionName
      batch,cycles,flops
      16,67042.96,132096
      ...
    Returns a dict mapping FunctionName -> DataFrame.
    """
    sections = {}
    with open(csv_file, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # skip blank lines
        if not line:
            i += 1
            continue

        # if this line has no commas, it's a section header
        if ',' not in line:
            func = line
            i += 1
            # next line must be the header row
            header = lines[i].strip()
            cols = header.split(',')
            i += 1

            buf = [header + '\n']
            # gather all data rows until blank or next header
            while i < len(lines) and lines[i].strip() and ',' in lines[i]:
                buf.append(lines[i])
                i += 1

            df = pd.read_csv(io.StringIO(''.join(buf)))
            sections[func] = df
        else:
            # stray line? skip
            i += 1

    return sections

def plot_performance(csv_file, title, out_name):
    # load each function's data
    data = load_sections(csv_file)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(color="lightgray", linestyle="-", linewidth=1, alpha=0.8)

    # plot each function
    for func, df in data.items():
        df = df.sort_values('batch')
        perf = df['flops'] / df['cycles']
        ax.plot(df['batch'], perf,
                marker='o', linestyle='-',
                label=func,
                linewidth=1.5)

    ax.set_xscale('log', base=2)
    ax.set_xticks(sorted({b for df in data.values() for b in df['batch']}))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Batch size", fontsize=12, labelpad=10)

    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold",
        loc="center",
        pad=20
    )
    ax.set_ylabel("FLOPs per cycle", fontsize=12)

    ax.legend(title="Function", loc="upper left", bbox_to_anchor=(1,1))

    plt.subplots_adjust(bottom=0.15, right=0.75)
    plt.savefig(out_name, dpi=300,
                bbox_inches="tight",
                facecolor=fig.patch.get_facecolor())
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <csv_file> <plot_title> <out_png>")
        sys.exit(1)

    _, csv_file, title, out_png = sys.argv
    plot_performance(csv_file, title, out_png)
