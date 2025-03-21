import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from GrOp_data_clean import data_clean

dir_path = 'clean_data/'
plot_path = 'plots/'
# minimum eigenvalue (solution) for every number of variables (4,6,8,9,10)
VARIABLES = [4,6,8,9,10]
SOLUTION_EIG = {4: -7.0, 6: -15.0, 8: -14.0, 9: -23.0, 10: -18.0}

def main(verbose = False):
    data_clean()
    heatmap_data = np.zeros((5, 10), float)
    with open(f'simulations_results.txt', 'w') as outfile:
        outfile.write('')

    # loop for every number of variables
    fig_iter_acc, axs_iter_acc = plt.subplots()
    fig_iter_time, axs_iter_time = plt.subplots()
    fig_enc_acc, axs_enc_acc = plt.subplots()
    fig_enc_time, axs_enc_time = plt.subplots()

    for i in VARIABLES:
        with open(f'simulations_results.txt', 'a') as outfile:
            outfile.write(f'Variables: {i}\n')
        try:
            # import data and get unique values of number of iterations
            df_iter = pd.read_csv(f'{dir_path}clean_simulations_{i}var_iterations.csv')
            number_of_iterations = df_iter['number of iterations'].unique()
            accuracies_iter = []
            times_iter = []
            if verbose:
                print(f'\n{i} variables with number of iterations: {number_of_iterations}')
            for n_iter in number_of_iterations:
                # for every unique number of iteration get a sub-dataframe
                df_n_iter = df_iter.loc[df_iter['number of iterations'] == n_iter]
                # get the mean execution time for that number of iteration
                mean_time = float('{:.3f}'.format(df_n_iter['time'].mean()))
                times_iter.append(mean_time)
                # get the counts of the eigenvalues
                eigenvalues_counts = df_n_iter['eigenvalue'].value_counts()
                # sum the total counts
                sum_eigenvalues = sum(eigenvalues_counts)
                # calculate the accuracy as counts of correct solution found over total counts
                accuracy = 0.0
                if SOLUTION_EIG[i] in eigenvalues_counts.keys():
                    accuracy = round(float(eigenvalues_counts[SOLUTION_EIG[i]] / sum_eigenvalues), 3)
                accuracies_iter.append(accuracy)

                if i == 10 and n_iter == 9:
                    heatmap_data[VARIABLES.index(i), i-2] = accuracy
                with open(f'simulations_results.txt', 'a') as outfile:
                    outfile.write(f'#Encoding Qubits: {i} ,#Iterations: {n_iter}, Accuracy: {accuracy}, Time (s): {mean_time}\n')

            if verbose:
                print(f'accuracy: {accuracies_iter}')
                print(f'time: {times_iter}')

            # create plots of accuracy and time vs #iterations
            axs_iter_acc.plot(number_of_iterations.astype('str'), accuracies_iter, '-o', label=f'{i} var')
            axs_iter_acc.set_title(f'Accuracy vs #Iterations, #Encoding Qubits = #var', fontsize = 10)
            axs_iter_acc.set(xlabel='#Iterations', ylabel='Accuracy')
            axs_iter_acc.legend(title='#Variables', loc='upper left')

            axs_iter_time.plot(number_of_iterations.astype('str'), times_iter, '-o', label=f'{i} var')
            axs_iter_time.set_title(f'Execution Time (s) vs #Iterations, #Encoding Qubits = #var', fontsize = 10)
            axs_iter_time.set(xlabel='#Iterations', ylabel='Time (s)')
            axs_iter_time.set_yscale('log')
            axs_iter_time.legend(title='#Variables', loc='upper left')

            # fig_iter.savefig(f"{plot_path}GrOp_{i}var_iter.png", dpi=300)
        except Exception as e:
            print(f'Exception: {e}')


        try:
            # import data and get unique values of encoding qubits
            df_enc = pd.read_csv(f'{dir_path}clean_simulations_{i}var_9iter_enc_qubit.csv')
            encoding_qubits = df_enc['encoding qubits'].unique()
            accuracies_enc = []
            times_enc = []
            if verbose:
                print(f'\n{i} variables with encoding qubits: {encoding_qubits}')
            for n_enc in encoding_qubits:
                # for every unique number of encoding qubits get a sub-dataframe
                df_n_enc = df_enc.loc[df_enc['encoding qubits'] == n_enc]
                # get the mean execution time for that number of encoding qubits
                mean_time = float('{:.3f}'.format(df_n_enc['time'].mean()))
                times_enc.append(mean_time)
                # get the counts of the eigenvalues
                eigenvalues_counts = df_n_enc['eigenvalue'].value_counts()
                # sum the total counts
                sum_eigenvalues = sum(eigenvalues_counts)
                # calculate the accuracy as counts of correct solution found over total counts
                accuracy = 0.0
                if SOLUTION_EIG[i] in eigenvalues_counts.keys():
                    accuracy = round(float(eigenvalues_counts[SOLUTION_EIG[i]] / sum_eigenvalues), 3)
                accuracies_enc.append(accuracy)


                heatmap_data[VARIABLES.index(i), n_enc - 2] = accuracy
                with open(f'simulations_results.txt', 'a') as outfile:
                    outfile.write(f'#Encoding Qubits: {n_enc}, #Iterations: 9, Accuracy: {accuracy}, Time (s): {mean_time}\n')

            if verbose:
                print(f'accuracy: {accuracies_enc}')
                print(f'time: {times_enc}')

            # create plots of accuracy and time vs #encoding qubits

            axs_enc_acc.plot(encoding_qubits.astype('str'), accuracies_enc, '-o', label=f'{i} var')
            axs_enc_acc.set_title(f'Accuracy vs #Encoding qubits, #Iterations: 9', fontsize = 10)
            axs_enc_acc.set(xlabel='#Encoding qubits', ylabel='Accuracy')
            axs_enc_acc.legend(title='#Variables', loc='upper left')

            axs_enc_time.plot(encoding_qubits.astype('str'), times_enc, '-o', label=f'{i} var')
            axs_enc_time.set_title(f'Time (s) vs #Encoding qubits, #Iterations: 9', fontsize = 10)
            axs_enc_time.set(xlabel='#Encoding qubits', ylabel='Time (s)')
            axs_enc_time.set_yscale('log')
            axs_enc_time.legend(title='#Variables', loc='upper left')

            # fig_enc.savefig(f"{plot_path}GrOp_{i}var_enc.png", dpi=300)
        except Exception as e:
            print(f'Exception: {e}')

    fig_enc_acc.savefig(f"{plot_path}GrOp_acc_vs_enc.png", dpi=300)
    fig_enc_time.savefig(f"{plot_path}GrOp_time_vs_enc.png", dpi=300)
    fig_iter_acc.savefig(f"{plot_path}GrOp_acc_vs_iter.png", dpi=300)
    fig_iter_time.savefig(f"{plot_path}GrOp_time_vs_iter.png", dpi=300)

    plt.figure()
    sns.heatmap(heatmap_data,
                annot = True,
                cmap = "RdYlGn",
                fmt = ".2f",
                xticklabels = range(2,12),
                yticklabels = VARIABLES,
                mask = np.isnan(heatmap_data))
    plt.title('Accuracy Heatmap by Variables and Encoding Qubits')
    plt.xlabel('#Encoding Qubits')
    plt.ylabel('#Variables')
    plt.savefig(f"{plot_path}GrOp_accuracy_heatmap.png", dpi=300)
    plt.close()

if __name__ == '__main__':
    main(verbose = False)