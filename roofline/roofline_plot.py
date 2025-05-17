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

class RooflinePlot:
    def __init__(self):
        pass

    def _load_data(self, csv_path: str):
        try:
            df = pd.read_csv(csv_path, skipinitialspace=True) 

            # Debugging
            print("DataFrame Columns:", df.columns)
            if not df.empty:
                print("Sample Row (first row):")
                print(df.head(1))
            else:
                print("DataFrame is empty, no sample row to print.")
                return None
            
            # Verify columns
            required_columns = ['flops', 'bytes_read', 'bytes_written', 'cycles', 'batch']
            for col in required_columns:
                if col not in df.columns:
                    print(f"Error: Required column '{col}' not found in CSV.")
                    print(f"Available columns: {list(df.columns)}")
                    return None

            # Calculate Operational Intensity and Performance
            # OI = flops / (bytes_read + bytes_written)
            df['operational_intensity'] = df['flops'] / (df['bytes_read'] + df['bytes_written'])
            # Performance = flops / cycles
            df['performance'] = df['flops'] / df['cycles']
            
            # Process data for plotting
            plot_data = []
            for index, row in df.iterrows():
                plot_data.append((f"Batch {row['batch']}", row['operational_intensity'], row['performance']))
            
            return plot_data
        
        # Error handling
        except FileNotFoundError:
            print(f"Error: CSV file not found at {csv_path}")
            return None
        except KeyError as e:
            print(f"Error: A required column was not found during data processing. Missing key: {e}")
            if 'df' in locals() and hasattr(df, 'columns'):
                 print(f"Available columns were: {list(df.columns)}")
            return None
        except Exception as e:
            print(f"Error loading or processing CSV: {e}")
            return None

    def plot(self, csv_path: str):
        data = self._load_data(csv_path)

        # From the datasheet: https://www.agner.org/optimize/instruction_tables.pdf (pg 364)
        peak_performance = 4  # flops/cycle (Scalar Peak π)
        avx2_peak_performance = 32  # flops/cycle (AVX2 Vector Peak π)
        memory_bandwidth = 53  # bytes/cycle (Memory Bandwidth β)


        plt.figure(figsize=(12, 9)) 
        plt.xscale('log')
        plt.yscale('log')

        # Set the x-axis range to the range of the data
        op_intensity = np.logspace(-3, 3, 100)
        
        # 1. Plot Memory Bandwidth Line (β)
        memory_bound_performance = op_intensity * memory_bandwidth
        plt.plot(op_intensity, memory_bound_performance, color='blue', linestyle='-', linewidth=2)
        text_oi_mem = 0.1
        text_perf_mem = text_oi_mem * memory_bandwidth
        
        plt.text(text_oi_mem, text_perf_mem, f'Mem BW (β) = {memory_bandwidth} B/cycle', 
                 color='blue', rotation=np.degrees(np.arctan(memory_bandwidth * (plt.gca().get_data_ratio()))), 
                 ha='left', va='bottom', rotation_mode='anchor', 
                 transform_rotates_text=True)

        # 2. Plot Scalar Peak Performance Line (π)
        plt.axhline(y=peak_performance, color='red', linestyle='--', linewidth=2)
        text_x_scalar = op_intensity[5] # A point on the x-axis, adjustable
        plt.text(text_x_scalar, peak_performance * 1.1, f'Scalar Peak (π) = {peak_performance} FLOPS/cycle', 
                 color='red', va='bottom', ha='left')

        # 3. Plot AVX2 Vector Peak Performance Line (Vector π)
        plt.axhline(y=avx2_peak_performance, color='green', linestyle='--', linewidth=2)
        text_x_avx2 = op_intensity[5] # A point on the x-axis, adjustable
        plt.text(text_x_avx2, avx2_peak_performance * 1.1, f'AVX2 Peak (Vector π) = {avx2_peak_performance} FLOPS/cycle', 
                 color='green', va='bottom', ha='left')

        # Plot title, labels and grid
        plt.xlabel('Operational Intensity (FLOPS/Byte)')
        plt.ylabel('Performance (FLOPS/Cycle)')
        plt.title('Roofline Plot for Intel Core i5-1035G4')
        plt.grid(True, which="major", ls="-", alpha=0.5, linewidth=0.8) # Apply to major ticks only
        
        # Plot the data points
        if data:
            all_ois_data_raw = [d[1] for d in data]
            all_perfs_data_raw = [d[2] for d in data]
            
            valid_data_points = [(label, oi, perf) for label, oi, perf in data if oi > 0 and perf > 0]

            if not valid_data_points and data: 
                 print("Some data points were filtered out due to non-positive values, not suitable for log scale.")
            
            if valid_data_points:
                # Extract OIs and Perfs for plotting the connecting line and markers
                plot_ois = [d[1] for d in valid_data_points]
                plot_perfs = [d[2] for d in valid_data_points]

                # 1. Plot the connecting dotted line (if there's more than one point)
                if len(valid_data_points) > 1:
                    plt.plot(plot_ois, plot_perfs, linestyle=':', color='dimgray', linewidth=1.2, alpha=0.7) # No label for legend

                # 2. Plot the individual markers with their labels on top of the line
                for kernel_name, oi_val, perf_val in valid_data_points:
                    # Each point is plotted individually to ensure it gets its unique label in the legend
                    plt.plot(oi_val, perf_val, 'o', markersize=8, label=kernel_name) 
            

            # Scaling of the roofline plot to the data points
            min_oi_data = min(all_ois_data_raw) if all_ois_data_raw else op_intensity.min()
            max_oi_data = max(all_ois_data_raw) if all_ois_data_raw else op_intensity.max()
            min_perf_data = min(all_perfs_data_raw) if all_perfs_data_raw else peak_performance / 10
            max_perf_data = max(all_perfs_data_raw) if all_perfs_data_raw else avx2_peak_performance
            
            current_xlim_min = min(op_intensity.min(), min_oi_data / 2 if min_oi_data > 0 else op_intensity.min())
            current_xlim_max = max(op_intensity.max(), max_oi_data * 2 if max_oi_data > 0 else op_intensity.max())
            
            current_ylim_min = min(peak_performance / 100, min_perf_data / 2 if min_perf_data > 0 else peak_performance / 100) 
            current_ylim_max = max(avx2_peak_performance * 2, max_perf_data * 2 if max_perf_data > 0 else avx2_peak_performance * 2)

            plt.xlim(max(1e-4, current_xlim_min), current_xlim_max) 
            plt.ylim(max(1e-4, current_ylim_min), current_ylim_max) 

        else:
            print("No data points to plot or all data points were invalid for log scale. Displaying roofs only.")
            plt.xlim(op_intensity.min(), op_intensity.max())
            min_y_val_roof_only = memory_bandwidth * op_intensity.min() 
            plt.ylim(max(1e-4, min_y_val_roof_only /10), avx2_peak_performance * 2)

        # Legend
        if data and valid_data_points: # Only show legend if there are plottable data points
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
            plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout only if legend is present
        else:
            plt.tight_layout()


        plt.show()


