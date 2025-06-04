# === Testbench CPU Configuration ===
# Parameter | Value
# Processor manufacturer, name, number, microarchitecture | Intel Core i5-1035G4, Ice Lake
# CPU Base Frequency | 1.10 GHz (nominal)
# CPU Maximum Frequency, Frequency Scaling support | 3.70 GHz, supports Turbo Boost
# Phase in Intel's development model | Tick (Ice Lake is a process shrink)

# Datasheet: https://www.intel.com/content/www/us/en/products/sku/196591/intel-core-i51035g4-processor-6m-cache-up-to-3-70-ghz/specifications.html
# Max Memory Bandwidth: 58.3 GB/s. At base freq of 1.10 Ghz, this is 58.3 GB / 1.10 GHz = 53 bytes / cycle
# 4 cores, 8 threads
# Supports AVX2, AVX-512

# From https://www.agner.org/optimize/instruction_tables.pdf (pg 364), IceLake has an VFMA throughput of 2 ops/cycle i.e. 4 flops/cycle
# With AVX2 (256-bit) instructions, this goes up to 32 flops/cycle for floating point operations (and 64 flops/cycle with AVX-512)

# === Roofline Plot ===
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from io import StringIO # Added for _load_data
import argparse      # Added for CLI

class RooflinePlot:
    def __init__(self):
        pass

    def _load_data(self, csv_path: str):
        try:
            with open(csv_path, 'r') as f:
                content = f.read()
        except FileNotFoundError:
            print(f"Error: CSV file not found at {csv_path}")
            return None
        except Exception as e:
            print(f"Error reading file {csv_path}: {e}")
            return None

        sections_data_strings = {} 
        current_section_name = None
        current_section_lines_for_df = []
        lines = content.splitlines()

        for line_content in lines:
            line = line_content.strip()
            if not line:
                continue

            is_section_header = (',' not in line) and (not line.lower().startswith("batch,cycles"))

            if is_section_header:
                if current_section_name and current_section_lines_for_df:
                    sections_data_strings[current_section_name] = "\n".join(current_section_lines_for_df)
                current_section_name = line
                current_section_lines_for_df = []
            elif line.lower().startswith("batch,cycles"):
                if not current_section_name: 
                    current_section_name = "Default Data" 
                current_section_lines_for_df.append(line)
            elif current_section_name:
                current_section_lines_for_df.append(line)

        if current_section_name and current_section_lines_for_df:
            sections_data_strings[current_section_name] = "\n".join(current_section_lines_for_df)

        if not sections_data_strings:
            print(f"No data sections could be parsed from {csv_path}.")
            return None

        processed_plot_data_map = {}
        required_columns = ['flops', 'bytes_read', 'bytes_written', 'cycles', 'batch']

        for name, csv_string_data in sections_data_strings.items():
            print(f"Processing section: {name}")
            try:
                df = pd.read_csv(StringIO(csv_string_data), skipinitialspace=True)
                df.columns = [col.strip() for col in df.columns]

                if df.empty:
                    print(f"DataFrame for section '{name}' is empty after parsing CSV string.")
                    continue
                
                missing_cols = [col for col in required_columns if col not in df.columns]
                if missing_cols:
                    print(f"Error: Section '{name}' - Required column(s) {missing_cols} not found.")
                    print(f"Available columns: {list(df.columns)}")
                    continue

                numeric_cols = ['flops', 'bytes_read', 'bytes_written', 'cycles']
                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df.dropna(subset=numeric_cols, inplace=True)
                if df.empty:
                    print(f"Section '{name}' has no valid numeric data after conversion/cleaning.")
                    continue
                
                # Calculate Operational Intensity and Performance
                # OI = flops / bytes_read (ignoring bytes_written as per user request)
                df['operational_intensity'] = np.where(df['bytes_read'] == 0, 0, df['flops'] / df['bytes_read'])
                # Performance = flops / cycles
                df['performance'] = np.where(df['cycles'] == 0, 0, df['flops'] / df['cycles'])
                
                section_points = []
                for index, row in df.iterrows():
                    section_points.append((row['operational_intensity'], row['performance']))
                
                if section_points:
                    processed_plot_data_map[name] = section_points
                else:
                    print(f"No valid data points generated for section '{name}'.")

            except pd.errors.EmptyDataError:
                print(f"Warning: No data to parse for section '{name}' (empty or header only). Skipping.")
                continue
            except Exception as e:
                print(f"Error processing section '{name}': {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if not processed_plot_data_map:
            print("No data successfully processed from any section.")
            return None
            
        return processed_plot_data_map

    def plot(self, csv_path: str):
        datasets_map = self._load_data(csv_path)

        # From the datasheet: https://www.agner.org/optimize/instruction_tables.pdf (pg 364)
        peak_performance = 4  # flops/cycle (Scalar Peak π)
        avx2_peak_performance = 32  # flops/cycle (AVX2 Vector Peak π)
        memory_bandwidth = 53  # bytes/cycle (Memory Bandwidth β)

        plt.figure(figsize=(12, 9)) 
        plt.xscale('log')
        plt.yscale('log')

        # Set the x-axis range for roofline visualization
        op_intensity = np.logspace(-3, 3, 100) # Default range, will be adjusted
        
        # 1. Plot Memory Bandwidth Line (β)
        memory_bound_performance = op_intensity * memory_bandwidth
        plt.plot(op_intensity, memory_bound_performance, color='blue', linestyle='-', linewidth=2)
        text_oi_mem = 0.1 # Position for the memory bandwidth label
        text_perf_mem = text_oi_mem * memory_bandwidth
        
        # Adjust rotation calculation if gca().get_data_ratio() causes issues or for simplicity
        try:
            rotation_angle = np.degrees(np.arctan(memory_bandwidth * (plt.gca().get_data_ratio_log() if hasattr(plt.gca(), 'get_data_ratio_log') else plt.gca().get_data_ratio() )))
        except Exception:
            rotation_angle = np.degrees(np.arctan(memory_bandwidth)) # Fallback simple rotation

        plt.text(text_oi_mem, text_perf_mem, f'Mem BW (β) = {memory_bandwidth} B/cycle', 
                 color='blue', rotation=rotation_angle, 
                 ha='left', va='bottom', rotation_mode='anchor', 
                 transform_rotates_text=True)

        # 2. Plot Scalar Peak Performance Line (π)
        plt.axhline(y=peak_performance, color='red', linestyle='--', linewidth=2)
        text_x_scalar = op_intensity[5] 
        plt.text(text_x_scalar, peak_performance * 1.1, f'Scalar Peak (π) = {peak_performance} FLOPS/cycle', 
                 color='red', va='bottom', ha='left')

        # 3. Plot AVX2 Vector Peak Performance Line (Vector π)
        plt.axhline(y=avx2_peak_performance, color='green', linestyle='--', linewidth=2)
        text_x_avx2 = op_intensity[5] 
        plt.text(text_x_avx2, avx2_peak_performance * 1.1, f'AVX2 Peak (Vector π) = {avx2_peak_performance} FLOPS/cycle', 
                 color='green', va='bottom', ha='left')

        plt.xlabel('Operational Intensity (FLOPS/Byte)')
        plt.ylabel('Performance (FLOPS/Cycle)', rotation='horizontal', ha='right')
        plt.title('Intel core i5-1035g4 @ 3.70GHz.\nCompiler: gcc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0')
        plt.grid(True, which="major", ls="-", alpha=0.5, linewidth=0.8)
        
        all_ois_data_raw = []
        all_perfs_data_raw = []
        plotted_anything = False

        if datasets_map:
            for section_name, points_list in datasets_map.items():
                if not points_list:
                    print(f"No points to plot for section: {section_name}")
                    continue

                valid_points_for_section = [(oi, perf) for oi, perf in points_list if oi > 0 and perf > 0]

                if not valid_points_for_section:
                    print(f"Section '{section_name}': All data points filtered out (non-positive OI/Perf for log scale).")
                    continue
                
                plot_ois = [p[0] for p in valid_points_for_section]
                plot_perfs = [p[1] for p in valid_points_for_section]

                all_ois_data_raw.extend(plot_ois)
                all_perfs_data_raw.extend(plot_perfs)
                
                plt.plot(plot_ois, plot_perfs, 'o', markersize=8, label=section_name)
                plotted_anything = True
            
            if all_ois_data_raw and all_perfs_data_raw:
                min_oi_data = min(all_ois_data_raw)
                max_oi_data = max(all_ois_data_raw)
                min_perf_data = min(all_perfs_data_raw)
                max_perf_data = max(all_perfs_data_raw)
                
                current_xlim_min = min(op_intensity.min(), min_oi_data / 2 if min_oi_data > 0 else op_intensity.min())
                current_xlim_max = max(op_intensity.max(), max_oi_data * 2 if max_oi_data > 0 else op_intensity.max())
                
                current_ylim_min = min(peak_performance / 100, min_perf_data / 2 if min_perf_data > 0 else peak_performance / 100) 
                current_ylim_max = max(avx2_peak_performance * 2, max_perf_data * 2 if max_perf_data > 0 else avx2_peak_performance * 2)

                plt.xlim(max(1e-4, current_xlim_min), current_xlim_max) 
                plt.ylim(max(1e-4, current_ylim_min), current_ylim_max) 
            elif plotted_anything: # Data was plotted, but maybe all_ois/perfs ended up empty after some logic error (should not happen if plotted_anything is true)
                 print("Warning: Data points were plotted, but auto-scaling based on data failed. Using default roof bounds.")
                 plt.xlim(op_intensity.min(), op_intensity.max())
                 min_y_val_roof_only = memory_bandwidth * op_intensity.min() 
                 plt.ylim(max(1e-4, min_y_val_roof_only / 10), avx2_peak_performance * 2)
            # If not plotted_anything, the 'else' block below handles limits.

        if not plotted_anything: # Covers datasets_map is None, empty, or all points filtered out
            print("No valid data points to plot. Displaying roofs with default bounds.")
            plt.xlim(op_intensity.min(), op_intensity.max())
            min_y_val_roof_only = memory_bandwidth * op_intensity.min() 
            plt.ylim(max(1e-4, min_y_val_roof_only / 10), avx2_peak_performance * 2)

        if plotted_anything:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
            plt.tight_layout(rect=[0, 0, 0.85, 1]) 
        else:
            plt.tight_layout()

        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a Roofline plot from a CSV file.")
    parser.add_argument("csv_path", type=str, help="Path to the CSV file containing performance data.")
    
    args = parser.parse_args()
    
    plotter = RooflinePlot()
    plotter.plot(args.csv_path)


