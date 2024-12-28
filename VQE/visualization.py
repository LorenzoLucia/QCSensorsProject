import matplotlib.pyplot as plt
import pandas as pd

results = pd.read_csv('results_rep.csv')
# Group by 'ansatz' and 'entanglement' and calculate the mean and standard deviation of 'accuracy'
grouped_results = results.groupby(['ansatz', 'entanglement'])['accuracy'].mean().reset_index()

# Plotting the results as a bar chart
fig, ax = plt.subplots()
width = 0.35  # the width of the bars

entanglements = grouped_results['entanglement'].unique()
ansatzes = grouped_results['ansatz'].unique()

x = range(len(entanglements))
for i, ansatz in enumerate(ansatzes):
    ansatz_data = grouped_results[grouped_results['ansatz'] == ansatz]
    ax.bar([p + width * i for p in x], ansatz_data['accuracy'], width, label=ansatz)
ax.set_xticks([p + width * (len(ansatzes) - 1) / 2 for p in x])
ax.set_xticklabels(entanglements)

ax.set_xlabel('Entanglement')
ax.set_ylabel('Accuracy')
ax.set_title('VQE Accuracy Results')
ax.legend(title='Ansatz')
ax.grid(True)
plt.show()

# Plotting the results
# fig, ax = plt.subplots()
# for key, grp in grouped_results.groupby(['ansatz']):
#     ax.errorbar(grp['entanglement'], grp['accuracy'], fmt='o', label=key, capthick=2, capsize=4)

# ax.set_xlabel('Entanglement')
# ax.set_ylabel('Accuracy')
# ax.set_title('VQE Accuracy Results')
# ax.legend(title='Ansatz')
# ax.grid(True)
# plt.show()

# grouped_results = results.groupby(['ansatz', 'entanglement']).agg({'accuracy': ['mean', 'std'], 'avg_time': 'mean'}).reset_index()

# Plotting the results as a bar chart
grouped_results = results.groupby(['ansatz', 'entanglement'])['avg_time'].mean().reset_index()
fig, ax = plt.subplots()
width = 0.35  # the width of the bars

entanglements = grouped_results['entanglement'].unique()
ansatzes = grouped_results['ansatz'].unique()

x = range(len(entanglements))
for i, ansatz in enumerate(ansatzes):
    ansatz_data = grouped_results[grouped_results['ansatz'] == ansatz]
    ax.bar([p + width * i for p in x], ansatz_data['avg_time'], width, label=ansatz)
ax.set_xticks([p + width * (len(ansatzes) - 1) / 2 for p in x])
ax.set_xticklabels(entanglements)

ax.set_xlabel('Entanglement')
ax.set_ylabel('Average Time')
ax.set_title('VQE Average Time Results')
ax.legend(title='Ansatz')
ax.grid(True)
plt.show()
# grouped_results.columns = ['ansatz', 'entanglement', 'accuracy_mean', 'accuracy_std', 'avg_time_mean']

# Plotting the results
# fig, ax = plt.subplots()
# for key, grp in grouped_results.groupby(['ansatz']):
#     ax.errorbar(grp['entanglement'], grp['avg_time'], fmt='o', label=key, capthick=2, capsize=4)

# ax.set_xlabel('Entanglement')
# ax.set_ylabel('Time')
# ax.set_title('VQE Time Results')
# ax.legend(title='Ansatz')
# ax.grid(True)
# plt.show()

# Plotting accuracy vs time
plt.plot(results['avg_time'], results['accuracy'], 'o')

# Calculate and display the correlation coefficient
correlation = results['avg_time'].corr(results['accuracy'])
plt.text(0.00, 1.00, f'Correlation: {correlation:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
plt.xlabel('Time')
plt.ylabel('Accuracy')
plt.title('VQE Accuracy vs Time')
plt.grid(True)
plt.show()
