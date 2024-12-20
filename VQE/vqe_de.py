import sys
sys.path.append("../")
from qubo.qubo import Qubo
from models.customgraph import CustomGraph
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.translators import to_ising
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Estimator
from qiskit.circuit.library import RealAmplitudes
from qiskit.providers.basic_provider import BasicProvider
from qiskit_ibm_runtime import Session, QiskitRuntimeService, EstimatorV2 as Estimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_aer import Aer

from scipy.optimize import differential_evolution
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

# IBMProvider.save_account('')
# provider = IBMProvider()

def cost_func(params, ansatz, hamiltonian, estimator, cost_history_dict):
    """Return estimate of energy from estimator

    Parameters:
        params (ndarray): Array of ansatz parameters
        ansatz (QuantumCircuit): Parameterized ansatz circuit
        hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
        estimator (EstimatorV2): Estimator primitive instance
        cost_history_dict: Dictionary for storing intermediate results

    Returns:
        float: Energy estimate
    """
    pub = (ansatz, [hamiltonian], [params])
    result = estimator.run(pubs=[pub]).result()
    energy = result[0].data.evs[0]

    cost_history_dict["iters"] += 1
    cost_history_dict["prev_vector"] = params
    cost_history_dict["cost_history"].append(energy)
    print(f"Iters. done: {cost_history_dict['iters']} [Current cost: {energy}]")

    return energy

class StopOptimization(Exception):
    pass

def make_callback(ansatz, hamiltonian, estimator, cost_history_dict):
    def callback(func, intermediate_result):
        if cost_func(intermediate_result, ansatz, hamiltonian, estimator, cost_history_dict) <= -165:
            raise StopOptimization
        return intermediate_result
    return callback
    # def callback(intermediate_result):
    #         if cost_func(intermediate_result.x, ansatz, hamiltonian, estimator, cost_history_dict) <= -165:
    #             raise StopOptimization
    #     return callback

def main(entanglement='sca', reps=1, maxiter=100, method='Powell'):
    graph = CustomGraph(
        n_columns=3,
        n_sensor_rows=2,
        n_street_points_rows=3,
        max_sensors_radius=2,
    )
    n_sensors, n_street_points, edges = graph.get_data_for_qubo()
    qubo = Qubo(n_sensors, n_street_points, edges)

    qp = QuadraticProgram()

    for i in range(len(qubo.matrix)):
        qp.binary_var()

    # qp.minimize(constant=0, quadratic=qubo.get_matrix_dict())
    qp.minimize(constant=0, quadratic=qubo.get_matrix_dict())

    hamiltonian, offset = to_ising(qp)

    # optimizer = COBYLA(maxiter=1)
    # optimizer = differential_evolution()

    # service = QiskitRuntimeService()
    # backend = service.least_busy(operational=True, simulator=True)
    backend = BasicProvider().get_backend('basic_simulator')

    cost_history_dict = {
            "prev_vector": None,
            "iters": 0,
            "cost_history": [],
        }

    ansatz = RealAmplitudes(num_qubits=len(qubo.matrix), entanglement=entanglement, reps=reps)
    num_params = ansatz.num_parameters
    bounds = [(0 * np.pi, 2 * np.pi) for _ in range(ansatz.num_parameters)]
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)

    x0 = 2 * np.pi * np.random.random(num_params)

    ansatz_isa = pm.run(ansatz)
    hamiltonian_isa = hamiltonian.apply_layout(layout=ansatz_isa.layout)

    

    with Session(backend=backend) as session:
        estimator = Estimator(mode=session)
        estimator.options.default_shots = 10000
        callback = make_callback(ansatz_isa, hamiltonian_isa, estimator, cost_history_dict)
        try:
            res = differential_evolution(
                cost_func,
                bounds,
                args=(ansatz_isa, hamiltonian_isa, estimator, cost_history_dict),
                x0=x0,
                maxiter=1000,
                popsize=30,
                mutation=(0.5,1.5),
                atol=0.001,
                callback=callback
            )
        except StopOptimization:
                print("Ottimizzazione interrotta: raggiunto valore di terminazione")

    # with Session(backend=backend) as session:
    #     estimator = Estimator(mode=session)
    #     estimator.options.default_shots = 10000
    #     callback = make_callback(ansatz_isa, hamiltonian_isa, estimator, cost_history_dict)
    #     try:
    #         res = minimize(
    #             cost_func,
    #             x0,
    #             bounds=bounds,
    #             args=(ansatz_isa, hamiltonian_isa, estimator, cost_history_dict),
    #             method=method,
    #             callback=callback,
    #             options={'gtol':1e-7}
    #         )
    #     except StopOptimization:
    #         print("Ottimizzazione interrotta: raggiunto valore di terminazione")

    fig, ax = plt.subplots()
    ax.plot(range(cost_history_dict["iters"]), cost_history_dict["cost_history"])
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Cost")
    plt.draw()
    plt.show()

    optimal_params = res.x
    print(res)
    print('Parametri ottimali:', optimal_params)
    print('Energia minima:', res.fun)


    # backend = Aer.get_backend('qasm_simulator')
    backend = BasicProvider().get_backend('basic_simulator')
    qc = ansatz.assign_parameters(res.x).decompose()
    
    qc.measure_all()

    # transpiled_qc = transpile(qc, backend)
    # qobj = assemble(transpiled_qc)
    result = backend.run(qc).result()

    counts = result.get_counts()
    sensor_counts  = {}
    for k,v in counts.items():
        if k[len(k)-n_sensors:] not in sensor_counts.keys():
            sensor_counts[k[len(k)-n_sensors:]] = v
        else:
            sensor_counts[k[len(k)-n_sensors:]] += v
    print(sensor_counts)

    max_value = max(sensor_counts.values())
    for k in sensor_counts.keys():
        if sensor_counts[k] == max_value:
            max_key = k
            break
    print('Risultati delle misurazioni:', max_key, max_value)

    active_sensors = []
    for i in range(n_sensors):
        if max_key[-1-i] == '1':
            active_sensors.append(i)
    print(active_sensors)

    graph.add_active_sensors(active_sensors)

    graph.plot()


if __name__ == '__main__':
    main(entanglement='full', reps=1, method='COBYQA', maxiter=1000)
