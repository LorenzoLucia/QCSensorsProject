import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

VARIABLES = 8
N_STREET_POINTS = 4
N_SENSORS = 4

res_GAS_path = 'results_comparison_8var.csv'
res_QAOA_path = '../QAOA/results/res_file.csv'
res_VQE_path = '../VQE/results_rep_8var.csv'
plot_path = 'plots/'

# Take the data frames of the different methods
df_GAS = pd.read_csv(res_GAS_path)
df_QAOA = pd.read_csv(res_QAOA_path)
df_VQE = pd.read_csv(res_VQE_path)

# Accuracy comparison plot
# take accuracies of different methods
df_GAS_acc = df_GAS['Accuracy']
df_QAOA_acc = df_QAOA.loc[df_QAOA['n_sensors'] == N_SENSORS]['accuracy']
df_VQE_acc = df_VQE['accuracy']

# concatenate the accuracy dataframes
accuracies =  pd.concat([df_VQE_acc, df_QAOA_acc, df_GAS_acc])
# create the list to differentiate the methods
methods = ['VQE']*len(df_VQE_acc) + ['QAOA']*len(df_QAOA_acc) + ['GAS']*len(df_GAS_acc)
data_acc = {'Method': methods, 'Accuracy': accuracies}

# Time comparison plot
# take data of different methods
df_GAS_time = df_GAS['Time']
df_QAOA_time = df_QAOA.loc[df_QAOA['n_sensors'] == N_SENSORS]['exec_time']
df_VQE_time = df_VQE['avg_time']

# concatenate the time dataframes
times =  pd.concat([df_VQE_time, df_QAOA_time, df_GAS_time])
# create the list to differentiate the methods
methods = ['VQE']*len(df_VQE_time) + ['QAOA']*len(df_QAOA_time) + ['GAS']*len(df_GAS_time)
data_time = {'Method': methods, 'Accuracy': times}

# create accuracy plot
sns.boxplot(x="Method", y="Accuracy", data=data_acc, showfliers=False, palette='Set2')
plt.xlabel('Method')
plt.ylabel('Accuracy')
plt.title('Comparison of Accuracy: VQE vs QAOA vs GAS (#sensors=4, #street_points=4)', fontsize=9)
plt.savefig(f'{plot_path}allmethods_comparison_accuracy.png', bbox_inches='tight')
plt.show()

# create time plot
sns.boxplot(x="Method", y="Accuracy", data=data_time, showfliers=False, palette='Set2')
plt.xlabel('Method')
plt.ylabel('Execution time (s)')
plt.yscale('log')
plt.title('Comparison of Execution Time: VQE vs QAOA vs GAS (#sensors=4, #street_points=4)', fontsize=9)
plt.savefig(f'{plot_path}allmethods_comparison_time.png', bbox_inches='tight')
plt.show()