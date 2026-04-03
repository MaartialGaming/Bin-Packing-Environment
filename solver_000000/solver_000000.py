from solver_000000.abstract_solver import AbstractSolver
from solver_000000.additional_script import AdditionalScript

class solver_000000(AbstractSolver):

    def __init__(self, inst):
        super().__init__(inst)
        self.name = 'solver_000000'

    def solve(self):
        additional_script = AdditionalScript()
        additional_script.doNothing()

        for idx, row in self.inst.df_items.iterrows():
            self.sol['type_vehicle'].append(self.inst.df_vehicles.index[0])
            self.sol['idx_vehicle'].append(self.idx_vehicle)
            self.sol['id_item'].append(idx)
            self.sol['x_origin'].append(0)
            self.sol['y_origin'].append(0)
            self.sol['z_origin'].append(0)
            self.sol['orient'].append(row.allowedRotations[0])
            self.idx_vehicle += 1
        
        self.write_solution_to_file()
#cambiamento
