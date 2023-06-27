from utils import *
from logic import *
clauses_1 = ['P==>Q', 
           '(L & M) ==> P', 
           '(B & L) ==> M', 
           '(A & P) ==> L', 
           '(A & B) ==> L', 
           A, 
           B]
clauses_2 = ['A ==> C', 
           '(D & ~B) ==> A', 
           'C ==> E', 
           'D', 
           '~B'] ## Thứ 7, Chủ nhật (E)

clauses_3 = ['A', 
           'A ==> B', 
           '(B & ~D) ==> C', 
           '~D'] # The door was open (C)
 

definite_clauses_KB = PropDefiniteKB()
for clause in clauses_1:
    definite_clauses_KB.tell(expr(clause))
print(pl_fc_entails(definite_clauses_KB, expr('C'))) #forward chaining
#print(pl_bc_entails(definite_clauses_KB, expr('C'))) #backward chaining
