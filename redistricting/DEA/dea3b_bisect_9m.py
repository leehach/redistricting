# The following line is not needed in Python 3. The future is here.
#from __future__ import division # safety with double division
from pyomo.environ import *
from pyomo.opt import SolverFactory
from pyomo.opt import SolverStatus, TerminationCondition
import numpy as np
import pandas as pd

df_g = pd.read_csv("plans1_g.csv",header=None)
df_b = pd.read_csv("plans2_b.csv",header=None)
df_eff = pd.DataFrame(columns=["Plan_Number_ID","DEA_Eff"])

def makeDict(aDF):
    (M,N) = aDF.shape
    print(M,N)
    a = {}
    for row in range(M):
        for col in range(N):
            a[(row+1,col+1)] = aDF.iloc[row,col]
            
    return (M, N, a)

(MM_g, NN_g, a_g) = makeDict(df_g)
(MM_b, NN_b, a_b) = makeDict(df_b)
if MM_g != MM_b:
	print("Incompatible numbers of rows in good and bad plan files...\n");
	
else:
	MM = MM_g


def calcEfficiency(a_g,a_b, MM_g,NN_g,MM_b, NN_b,MM, iHat):
   #cur_LB = 0.001
   # The LB below based on theory of LB
   cur_LB = 0.001
   cur_UB = 1.0
   eps_tol = 0.0001
   count1 = 0
   DEAmod = ConcreteModel()
   S = NN_g + NN_b
   DEAmod.N_g = RangeSet(1,NN_g)
   DEAmod.N_b = RangeSet(1,NN_b)
   DEAmod.N = Set(initialize = DEAmod.N_g | DEAmod.N_b)
   DEAmod.M_g = RangeSet(1,MM_g)
   DEAmod.M_b = RangeSet(1,MM_b)
   DEAmod.M = Set(initialize = DEAmod.M_g | DEAmod.M_b)
   DEAmod.lbda = Var(DEAmod.M, within = NonNegativeReals, initialize = 1/MM_g, bounds = (0,1))
   DEAmod.eps = Var(within = NonNegativeReals, initialize = 0.0, bounds = (0,1))
   DEAmod.theta = Param(within = NonNegativeReals, initialize = 0.1, mutable = True)
            
        
   def obj_rule (DEAmod):
      return sum(DEAmod.lbda[i]for i in DEAmod.M)
   DEAmod.obj = Objective(rule = obj_rule, sense = maximize)
	  
   def con_smaller_rule(DEAmod,m):
      return sum(a_b[i,m]*DEAmod.lbda[i]for i in DEAmod.M) - (DEAmod.theta*a_b[iHat,m])  <= 0
   DEAmod.con_smaller = Constraint(DEAmod.N_b, rule = con_smaller_rule)
		
   def con_bigger_rule (DEAmod, m):
      return sum(a_g[i,m]*DEAmod.lbda[i] for i in DEAmod.M) - ((1/DEAmod.theta)*a_g[iHat,m]) >= 0
   DEAmod.con_bigger = Constraint(DEAmod.N_g, rule = con_bigger_rule)
	
   def con_lambdas_rule(DEAmod):
      return sum(DEAmod.lbda[i] for i in DEAmod.M) == 1
   DEAmod.con_lambdas = Constraint(rule = con_lambdas_rule)
   
   def con_eps_rule(DEAmod):
      return DEAmod.eps <= 1
   DEAmod.con_eps = Constraint(rule = con_eps_rule)
   
   
   Opt = SolverFactory("glpk")
   #DEAinstance = DEAmod.create_instance()
   # Above statement only for an Abstract Model??
   while (abs(cur_UB - cur_LB) > eps_tol):
      count1 = count1 + 1 
      cur_theta = (cur_LB + cur_UB)/2.0 
      #print("count1 = ",count1,"cur_theta = ",cur_theta)
	  # Create the concrete model and solve
      #DEAinstance = DEAmod.create_instance()
      DEAmod.theta = cur_theta
      #DEAinstance.theta = cur_theta
      
        
      #Soln = Opt.solve(DEAinstance)
      Soln = Opt.solve(DEAmod)
      #DEAmod.load(Soln)
      #DEAinstance.solutions.load_from(Soln)
      #DEAmod.solutions.load_from(Soln)
      if (Soln.solver.status == SolverStatus.ok) and (Soln.solver.termination_condition == TerminationCondition.optimal):
         cur_UB = cur_theta
         #for i in DEAmod.M:
            #print(i,value(DEAmod.obj),value(DEAmod.theta),value(DEAmod.lbda[i]))
      elif (Soln.solver.status == SolverStatus.ok) and (Soln.solver.termination_condition == TerminationCondition.infeasible):
         cur_LB = cur_theta
         #print("Solver status: ", Soln.solver.status)
         #print("Solver Termination Condition:  ", Soln.solver.termination_condition)
      else:
         #print("Solver status: ", Soln.solver.status)
         #print("Solver Termination Condition:  ", Soln.solver.termination_condition)
		 # IS THIS CORRECT?
         cur_LB = cur_theta
         #for i in DEAmod.M:
            #print("AT ELSE PART ",i,value(DEAmod.obj),value(DEAmod.theta),value(DEAmod.lbda[i]))
		 
   cur_theta = cur_UB
   print(cur_LB, cur_UB, cur_theta)
   return cur_theta
    

count = -1;
for iHat in range(1,MM+1):
	eff = calcEfficiency(a_g,a_b, MM_g,NN_g,MM_b, NN_b,MM, iHat)
	count += 1
	print("count = ", count)
	print("DMU "+str(iHat)+" has efficiency "+str(eff))
	df_eff = df_eff.append({'Plan_Number_ID': count, 'DEA_Eff': eff},ignore_index=True)
		
df_eff.to_csv('df_eff_2355_9m_bisect_eps0001_aug27.csv')





