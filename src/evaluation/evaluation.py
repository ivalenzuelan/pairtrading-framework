import json
import csv
import os
from collections import defaultdict

# Load your JSON data
with open('./results/backtest_results.json') as f:
    data = json.load(f)

# Create output directory
os.makedirs('csv_output', exist_ok=True)

# 1. Summary CSV
with open('csv_output/summary.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(data['summary'].keys())
    writer.writerow(data['summary'].values())

# 2. Method Performance CSV
with open('csv_output/method_performance.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    headers = ['method'] + list(next(iter(data['method_performance'].values())).keys())
    writer.writerow(headers)
    
    for method, metrics in data['method_performance'].items():
        row = [method] + [metrics[k] for k in headers[1:]]
        writer.writerow(row)

# 3. Method Comparison CSV
with open('csv_output/method_comparison.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    headers = ['method_pair', 't_statistic', 'p_value', 'significant_1%', 'significant_5%', 'significant_10%']
    writer.writerow(headers)
    
    for pair, comparison in data['method_comparison'].items():
        writer.writerow([
            pair,
            comparison['t_statistic'],
            comparison['p_value'],
            comparison['significant_1%'],
            comparison['significant_5%'],
            comparison['significant_10%']
        ])

# 4. Window Results CSV
with open('csv_output/window_results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    headers = list(data['detailed_results']['window_results'][0].keys())
    headers[headers.index('selected_pairs')] = 'selected_pair_1;selected_pair_2;selected_pair_3;selected_pair_4;selected_pair_5'
    writer.writerow(headers)
    
    for window in data['detailed_results']['window_results']:
        row = list(window.values())
        # Flatten selected pairs list
        if 'selected_pairs' in window:
            pairs = window['selected_pairs']
            row[row.index(pairs)] = ';'.join(pairs)
        writer.writerow(row)

# 5. Position Details CSV
with open('csv_output/position_details.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    headers = list(data['detailed_results']['position_details'][0].keys())
    headers[headers.index('training_methods')] = 'training_methods'
    writer.writerow(headers)
    
    for position in data['detailed_results']['position_details']:
        row = list(position.values())
        # Flatten training methods list
        if 'training_methods' in position:
            methods = position['training_methods']
            row[row.index(methods)] = ';'.join(methods)
        writer.writerow(row)

print("CSV files created successfully in 'csv_output' directory!")
