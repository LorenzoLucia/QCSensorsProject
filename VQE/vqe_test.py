import sys
sys.path.append("../")
from qubo.qubo import Qubo
from models.customgraph import CustomGraph
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.translators import to_ising
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA, POWELL
from qiskit.primitives import Estimator
from qiskit.circuit.library import RealAmplitudes


def entanglement_options_test():

    outfile = "entanglement_comparison_vqe.csv"

    fout = open(outfile, 'w')

    graph = CustomGraph(
        n_columns=3,
        n_sensor_rows=2,
        n_street_points_rows=4,
        max_sensors_radius=2,
    )
    n_sensors, n_street_points, edges = graph.get_data_for_qubo()
    qubo = Qubo(n_sensors, n_street_points, edges)

    qp = QuadraticProgram()

    for i in range(len(qubo.matrix)):
        qp.binary_var()

    qp.minimize(constant=0, quadratic=qubo.get_matrix_dict())

    operator, offset = to_ising(qp)
    # print('Hamiltoniana:', operator)
    # print('Offset:', offset)

    optimizer = POWELL(maxiter=100)
    fout.write("entanglement,1,2,3")
    ## linear entanglement
    fout.write("\nlinear")
    for i in range(1, 4):
        ansatz = RealAmplitudes(num_qubits=len(qubo.matrix), entanglement="linear", reps=i)
        vqe = VQE(ansatz=ansatz, optimizer=optimizer, estimator=Estimator())
        result = vqe.compute_minimum_eigenvalue(operator)
        fout.write(f",{result.eigenvalue.real}")

    ## full entanglement
    fout.write("\nfull")
    for i in range(1, 4):
        ansatz = RealAmplitudes(num_qubits=len(qubo.matrix), entanglement="full", reps=i)
        vqe = VQE(ansatz=ansatz, optimizer=optimizer, estimator=Estimator())
        result = vqe.compute_minimum_eigenvalue(operator)
        fout.write(f",{result.eigenvalue.real}")

    ## circular entanglement
    fout.write("\ncircular")
    for i in range(1, 4):
        ansatz = RealAmplitudes(num_qubits=len(qubo.matrix), entanglement="circular", reps=i)
        vqe = VQE(ansatz=ansatz, optimizer=optimizer, estimator=Estimator())
        result = vqe.compute_minimum_eigenvalue(operator)
        fout.write(f",{result.eigenvalue.real}")

    ## sca entanglement
    fout.write("\nsca")
    for i in range(1, 4):
        ansatz = RealAmplitudes(num_qubits=len(qubo.matrix), entanglement="sca", reps=i)
        vqe = VQE(ansatz=ansatz, optimizer=optimizer, estimator=Estimator())
        result = vqe.compute_minimum_eigenvalue(operator)
        fout.write(f",{result.eigenvalue.real}")

    ## reverse_linear entanglement
    fout.write("\nreverse_linear")
    for i in range(1, 4):
        ansatz = RealAmplitudes(num_qubits=len(qubo.matrix), entanglement="reverse_linear", reps=i)
        vqe = VQE(ansatz=ansatz, optimizer=optimizer, estimator=Estimator())
        result = vqe.compute_minimum_eigenvalue(operator)
        fout.write(f",{result.eigenvalue.real}")



if __name__ == '__main__':
    entanglement_options_test()