from constraint import *

problem = Problem()

# Without hueristic
problem.addVariables(["WA", "NT", "Q", "NSW", "V", "SA", "T"], ['R', 'G', 'B'])

# With hueristic
# problem.addVariable("SA", ['R'])
# problem.addVariables(["WA", "NT", "Q", "NSW", "V"], ['G', 'B'])
# problem.addVariable("T", ['R', 'G', 'B'])


def dif_func(a, b):
    return a != b


problem.addConstraint(dif_func, ["SA", "WA"])
problem.addConstraint(dif_func, ["SA", "NT"])
problem.addConstraint(dif_func, ["SA", "Q"])
problem.addConstraint(dif_func, ["SA", "NSW"])
problem.addConstraint(dif_func, ["SA", "V"])
problem.addConstraint(dif_func, ["WA", "NT"])
problem.addConstraint(dif_func, ["Q", "NT"])
problem.addConstraint(dif_func, ["Q", "NSW"])
problem.addConstraint(dif_func, ["V", "NSW"])

solutions = problem.getSolutions()

i = 1
for solution in solutions:
    print('Solution', i, ':', solution, '\n')
    i += 1
