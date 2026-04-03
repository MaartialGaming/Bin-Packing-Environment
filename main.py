from instances import Instance
from solver_000000 import solver_000000

if __name__ == '__main__':

    dataset_name = 'DatasetA'

    inst = Instance(dataset_name)

    solver = solver_000000(inst)

    solver.solve()

#aggiungo modifiche solo per capire il funzionamento di GH