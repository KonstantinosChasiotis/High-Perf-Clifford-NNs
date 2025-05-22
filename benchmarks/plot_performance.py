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
    fig.patch.set_facecolor('white')  # White overall figure background
    ax.set_facecolor('lightgrey')     # Gray background for the plotting area
    ax.grid(color="white", linestyle="-", linewidth=1, alpha=0.8) # White grid lines

    # plot each function
    for func, df in data.items():
        df = df.sort_values('batch')
        # Transform batch size to log2
        df['log2_batch'] = np.log2(df['batch'])
        perf = df['flops'] / df['cycles']
        line, = ax.plot(df['log2_batch'], perf,
                        marker='o', linestyle='-',
                        # label=func, # Removed for direct labeling
                        linewidth=1.5)
        # Add text label near the middle of the line, above it
        num_points = len(df['log2_batch'])
        middle_idx = num_points // 2
        x_pos = df['log2_batch'].iloc[middle_idx]
        y_pos = perf.iloc[middle_idx]
        
        # Adjust y_pos to be slightly above the line; this might need tweaking
        # depending on the typical y-axis scale and variation.
        y_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03 # 3% of y-axis range
        ax.text(x_pos, y_pos + y_offset, func, va='bottom', ha='center', color=line.get_color(), fontweight='bold')


    # ax.set_xscale('log', base=2) # Removed as data is already log2
    # Use unique log2 batch sizes for x-ticks
    unique_log2_batches = sorted({np.log2(b) for df_val in data.values() for b in df_val['batch']})
    ax.set_xticks(unique_log2_batches)
    # ax.get_xaxis().set_major_formatter(plt.ScalarFormatter()) # Keep default formatter for linear scale
    ax.set_xlabel("Batch size (2^N)", fontsize=12, labelpad=10)

    ax.set_title(
        "Intel core i5-1035g4 @ 3.70GHz\nCompiler: gcc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0",
        fontsize=14,
        fontweight="bold",
        loc="center",
        pad=20
    )
    ax.set_ylabel("FLOPs per cycle", fontsize=12)

    # ax.legend(title="Function", loc="upper left", bbox_to_anchor=(1,1)) # Removed legend

    plt.subplots_adjust(bottom=0.15, right=0.95, top=0.85) # Adjusted margins
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
