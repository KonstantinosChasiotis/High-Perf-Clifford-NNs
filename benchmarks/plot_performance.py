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

    label_pos = {} #hard code pos to avoid overlap
    label_pos["Clifford Group Norm"] = [(0, 0.2), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)] #[(x,y),....]
    label_pos["Multivector Activation"] = [(0, 0.2), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
    label_pos["Clifford Linear"] = [(0, 0.2), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]

    # load each function's data
    data = load_sections(csv_file)

    fig, ax = plt.subplots(figsize=(10, 6))

    # styling to match asl plots
    fig.patch.set_facecolor('white')
    ax.set_facecolor('lightgrey')
    ax.xaxis.grid(color="white", linestyle="-", linewidth=1, alpha=0.0)
    ax.yaxis.grid(color="white", linestyle="-", linewidth=2, alpha=0.8)

    for i, (func, df) in enumerate(data.items()):
        df = df.sort_values('batch')

        # plot in log2
        df['log2_batch'] = np.log2(df['batch'])
        perf = df['flops'] / df['cycles']
        line, = ax.plot(df['log2_batch'], perf,
                        marker='o', linestyle='-',
                        linewidth=2)
        
        # add in function name labels to the lines
        num_points = len(df['log2_batch'])
        middle_idx = num_points // 2
        x_pos = df['log2_batch'].iloc[middle_idx]
        y_pos = perf.iloc[middle_idx]
        #y_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03 # 3% of y-axis range
        y_offset = label_pos["Clifford Group Norm"][i][1]
        x_offset = label_pos["Clifford Group Norm"][i][0]
        ax.text(x_pos + x_offset, y_pos + y_offset, func, va='bottom', ha='center', color=line.get_color(), fontweight='bold')

    unique_log2_batches = sorted({np.log2(b) for df_val in data.values() for b in df_val['batch']})
    ax.set_xticks(unique_log2_batches)
    ax.set_xlabel("Batch size ($2^N$)", fontsize=12, labelpad=10)

    # hardcoded to benchmark machine
    ax.set_title(
        f"{title} on Intel core i5-1035g4 @ 3.70GHz\n ", #Compiler: gcc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0
        fontsize=14,
        fontweight="bold",
        loc="left",
        pad=20
    )

    fig.text(0.126, 0.91, 'Performance [FLOPs/cycle] vs. Batch size',
                 ha='left', va='top', fontsize=13, fontweight='normal')
    
    ax.set_ylabel("FLOPs / cycle", fontsize=12, rotation='horizontal', ha='right')

    for spine in ax.spines.values():
        spine.set_visible(False)

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
