import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

# Funzione per aggiungere virgole ai valori di params
def fix_params(param_str):
    fixed_str = re.sub(r'(\d)(\s+)(\d)', r'\1, \3', param_str)
    return eval(fixed_str)

# Load the data from the CSV file
file_path = "res_file.csv"
data = pd.read_csv(file_path)

# Correggi la colonna 'params'
data['params'] = data['params'].apply(lambda x: fix_params(x) if isinstance(x, str) else x)
data['final_config'] = data['final_config'].apply(lambda x: eval(x) if isinstance(x, str) else x)

# Configurazione generale per i grafici
sns.set(style="whitegrid")

# Palette dinamica personalizzata ad alto contrasto
base_colors = ["#FF0000", "#00AA00", "#0000FF", "#AA00FF"]  # Colori base
unique_sensors = sorted(data['n_sensors'].unique())  # Trova i valori univoci di n_sensors
custom_palette = {sensor: base_colors[i % len(base_colors)] for i, sensor in enumerate(unique_sensors)}

# Plot 1: Accuracy vs Circuit Depth (p) by the number of sensors
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=data,
    x="p",
    y="accuracy",
    hue="n_sensors",
    style="n_street_points",
    markers=True,
    palette=custom_palette,  # Usa la palette personalizzata
    linewidth=2.5,
    markersize=10
)
plt.title("Accuracy vs Circuit Depth (p)", fontsize=16)
plt.xlabel("Circuit Depth (p)", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.xticks(range(1, 6, 1), fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title="Sensors / Street Points", fontsize=10, title_fontsize=12, loc="best")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("accuracy_vs_p.png", dpi=300)
plt.close()

# Plot 2: Execution time vs Circuit Depth (p) by the number of sensors (log scale)
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=data,
    x="p",
    y="exec_time",
    hue="n_sensors",
    style="n_street_points",
    markers=True,
    palette=custom_palette,  # Usa la palette personalizzata
    linewidth=2.5,
    markersize=10
)
plt.title("Execution Time vs Circuit Depth (p)", fontsize=16)
plt.xlabel("Circuit Depth (p)", fontsize=14)
plt.ylabel("Execution Time (s)", fontsize=14)
plt.yscale("log")  # Scala logaritmica
plt.xticks(range(1, 6, 1), fontsize=12)
plt.yticks(fontsize=12)

# Sposta la legenda fuori dal grafico
plt.legend(
    title="Sensors / Street Points",
    fontsize=10,
    title_fontsize=12,
    loc="upper left",  # Posiziona la legenda in alto a sinistra
    bbox_to_anchor=(1, 1)  # Spostala fuori dal grafico (destra in alto)
)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("exec_time_vs_p.png", dpi=300, bbox_inches="tight")  # Usa bbox_inches per includere la legenda
plt.close()

# Plot 3: Accuracy vs Execution time (log scale for exec_time, con linee spezzate)
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=data,
    x="exec_time",
    y="accuracy",
    hue="n_sensors",
    style="n_street_points",
    markers=True,
    palette=custom_palette,  # Usa la palette personalizzata
    linewidth=2.5,
    markersize=10
)
plt.title("Accuracy vs Execution Time", fontsize=16)
plt.xlabel("Execution Time (s)", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.xscale("log")  # Scala logaritmica
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Sposta la legenda fuori dal grafico
plt.legend(
    title="Sensors / Street Points",
    fontsize=10,
    title_fontsize=12,
    loc="upper left",  # Posiziona la legenda in alto a sinistra
    bbox_to_anchor=(1, 1)  # Spostala fuori dal grafico (destra in alto)
)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("accuracy_vs_exec_time.png", dpi=300, bbox_inches="tight")  # Usa bbox_inches per includere la legenda
plt.close()
# Plot 4: Heatmap of Accuracy by p and n_sensors
heatmap_data = data.pivot_table(values="accuracy", index="n_sensors", columns="p", aggfunc="mean")
plt.figure(figsize=(10, 6))
sns.heatmap(
    heatmap_data,
    annot=True,
    cmap="RdYlGn",  # Scala cromatica rosso-giallo-verde per evidenziare i valori
    fmt=".2f",
    cbar_kws={'label': 'Accuracy'}
)
plt.title("Heatmap of Accuracy (n_sensors vs p)", fontsize=16)
plt.xlabel("Circuit Depth (p)", fontsize=14)
plt.ylabel("Number of Sensors", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig("accuracy_heatmap.png", dpi=300)
plt.close()
