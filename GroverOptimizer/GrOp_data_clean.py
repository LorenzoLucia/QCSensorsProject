import os
import fnmatch
import re

dir_path = 'raw_data/'
header_rot = 'encoding qubits,number of iterations,rotations,eigenvalue\n'
header_iter = 'encoding qubits,number of iterations,eigenvalue,time\n'
header_enc = 'encoding qubits,eigenvalue,time\n'

def data_clean():
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        #print(file_path)
        if fnmatch.fnmatch(filename, '*.csv'):
            if fnmatch.fnmatch(filename, 'simulations_*.csv'):
                with open(file_path, 'r') as infile:
                    with open(("clean_data/clean_" + filename), 'w') as outfile:
                        index = 0
                        for line in infile:
                            if line.startswith('e'):
                                if index == 0:
                                    if fnmatch.fnmatch(filename, '*rotations*'):
                                        line = header_rot
                                    elif fnmatch.fnmatch(filename, '*iterations*'):
                                        line = header_iter
                                    elif fnmatch.fnmatch(filename, '*enc*'):
                                        line = header_enc
                                    index += 1
                                else:
                                    line = '\n'
                            else:
                                line = line.replace('\t ', '')
                                line = line.replace('\t', '')
                                match_str = re.search(r'\{.*\},', line)
                                if match_str is not None:
                                    line = line.replace(match_str.group(0), '')
                            outfile.write(line)

def main():
    data_clean()

if __name__ == '__main__':
    main()