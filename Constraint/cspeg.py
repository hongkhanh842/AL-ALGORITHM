from constraint import *

problem = Problem()

problem.addVariable("A", [1, 2, 4])
problem.addVariable("B", [1, 3, 5])
problem.addVariable("C", [3, 5])
problem.addVariable("D", [0, 1, 2])

problem.addConstraint(lambda A, B: A > B, ["A", "B"])
problem.addConstraint(lambda C, B: C == B, ["C", "B"])
problem.addConstraint(lambda D, B: D * 2 < B, ["D", "B"])
problem.addConstraint(lambda D, C: D * D < C, ["D", "C"])

solutions = problem.getSolutions()

i = 1
for solution in solutions:
    print('Solution', i, ':', solution, '\n')
    i += 1
