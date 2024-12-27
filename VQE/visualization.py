import matplotlib.pyplot as plt
import pandas as pd

results = pd.read_csv('results_rep.csv')
# Group by 'ansatz' and 'entanglement' and calculate the mean and standard deviation of 'accuracy'
grouped_results = results.groupby(['ansatz', 'entanglement'])['accuracy'].mean().reset_index()

# Plotting the results
fig, ax = plt.subplots()
for key, grp in grouped_results.groupby(['ansatz']):
    ax.errorbar(grp['entanglement'], grp['accuracy'], fmt='o', label=key, capthick=2, capsize=4)

ax.set_xlabel('Entanglement')
ax.set_ylabel('Accuracy')
ax.set_title('VQE Accuracy Results')
ax.legend(title='Ansatz')
ax.grid(True)
plt.show()