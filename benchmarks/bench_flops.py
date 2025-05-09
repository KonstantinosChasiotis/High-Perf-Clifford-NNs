import numpy as np
import torch
import csv

from parse_benchconfig import Benchconfig
from cliffordlayers.wrapper_groupnorm import c_clifford_group_norm

def bench_flops_groupnorm(benchconf : Benchconfig):
    #config readout
    versions = benchconf.config["groupnorm_versions"]
    B = [int(b) for b in benchconf.config["groupnorm_B"]]
    C = int(benchconf.config["groupnorm_C"][0])
    D = int(benchconf.config["groupnorm_D"][0])
    I = int(benchconf.config["groupnorm_I"][0])
    num_groups = int(benchconf.config["groupnorm_num_groups"][0])

    #overwrite old files and add headers
    header = ["B", "flops"]
    for vers in versions:
        csvpath = f"benchmarks/flops/groupnorm_{vers}_flops.csv"
        try:
            with open(csvpath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header)
            print(f"CSV file '{csvpath}' created successfully with header: {', '.join(header)}")
        except Exception as e:
            print(f"An error occurred while creating the CSV file: {e}")

    for vers in versions:
        for b in B:
            print(f"benchflops.py: version={vers}, B={b}, C={C}, D={D}, I={I}, num_groups={num_groups}")
            x = torch.randn(b, C, D, I)

            x_norm_c = c_clifford_group_norm(
                f"{vers}_flops",
                x,
                I,
                num_groups=num_groups,
                running_mean=None,
                running_cov=None,
                weight=None,
                bias=None,
            )

def main():
    #load config
    benchconf = Benchconfig("benchmarks/bench.config")
    bench_flops_groupnorm(benchconf)

if __name__ == "__main__":
    main()
