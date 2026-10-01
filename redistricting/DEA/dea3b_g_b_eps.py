# The following line is not needed in Python 3. The future is here.
#from __future__ import division # safety with double division
from pyomo.environ import *
from pyomo.opt import SolverFactory
from pyomo.opt import SolverStatus, TerminationCondition
import numpy as np
import pandas as pd
import os

# Slacks-based measure of DEA
# What is it?

# A slack is a gap between one plan and better plan(s).

# ┌──────┬────────────────────┬─────────────────────┐
# │ Plan │ Compactness (good) │ County splits (bad) │
# ├──────┼────────────────────┼─────────────────────┤
# │ A    │ 0.40               │ 10                  │
# ├──────┼────────────────────┼─────────────────────┤
# │ B    │ 0.50               │ 6                   │
# └──────┴────────────────────┴─────────────────────┘

# Measured against B, A has two slacks. A slack says how much better a plan/row could be for a given metric.

# And SBM then?
# Created by Kaoru Tone in 2001, it is one way to score DEA.
# Turn slacks into percetnages of the plan's own value:
# compactness - 0.10 (slack) / 0.40 (actual value) = 25%
# splits - 4 / 10 -> 40%
# combine into one score -> bigger gaps give a lower score

# The above uses two plans, but when you have many plans to compare to, that's when you now need this model + solver.

# Now.. something I haven't quite nailed down, but I will, is the fact that tehre are 

#folder_path = "C:\w64"
#from folder_path import *

# inputs

# note no headers, so this is going off memory/investigation
# Idea: we can keep a dict of measures who are good and bad and dynamically lookup to avoid this.
df_g = pd.read_csv("plans1_g.csv",header=None) # metrics/critera for which "good is better" -> higher is better (i.e. compactness)
df_b = pd.read_csv("plans2_b.csv",header=None) # metrics/critera for whiich "bad is better" -> lower is better (i.e county splits)

# create dataframe to hold data
df_eff = pd.DataFrame(columns=["Plan_Number_ID","DEA_Eff"])

# formats each table into a 1-indexed dict - needed for pyomo
# What is pyomo? It's OSS Optimization Modeling, it models problems to be sent to an external solver.

def makeDict(aDF):
    (M,N) = aDF.shape
    print(M,N)
    a = {}
    for row in range(M):
        for col in range(N):
            a[(row+1,col+1)] = aDF.iloc[row,col] # iloc = integer location (pandas) / # plus 1 because the model below starts at one (RangeSet)
            
    return (M, N, a)

# What's happening here we're turning the csv data from this:

# 0.40,0.70
# 0.50,0.60
# 0.45,0.80

# to 

# M = 3
# N = 2
# a = {
#     (1, 1): 0.40,  (1, 2): 0.70,   # plan 1
#     (2, 1): 0.50,  (2, 2): 0.60,   # plan 2
#     (3, 1): 0.45,  (3, 2): 0.80,   # plan 3
# }

# M is rows and N is coluimns
    

(MM_g, NN_g, a_g) = makeDict(df_g)
(MM_b, NN_b, a_b) = makeDict(df_b)

# compare rows - ensure they're the same size...
# this also doesn't throw... which... probably is a good idea?
if MM_g != MM_b:
	print("Incompatible numbers of rows in good and bad plan files...\n");
	
else:
	MM = MM_g


# this scores one **plan** at a time, so that's row by row
# This accepts:
# the "good" and "bad" plans
# the shapes of each
# then `MM`` unused variable that sets the definitive shape above - we could use it and loop in the function or we'll see
# iHat is the row number

def calcEfficiency(a_g,a_b, MM_g,NN_g,MM_b, NN_b,MM, iHat):

   # Instantiate an empty pyomo model
   DEAmod = AbstractModel()
   
   # total number of metrics (these are columns)
   S = NN_g + NN_b

   # set good and bad metrics to properties on the model
   # interesting is that the attribute namne here doesn't matter - N_g for instance.
   # Pyomo will go by the types - the RangeSet or Set or Var
   # so this is all setup
   DEAmod.N_g = RangeSet(1,NN_g) # range of columns in good
   DEAmod.N_b = RangeSet(1,NN_b) # range of columns in bad
   DEAmod.N = Set(initialize = DEAmod.N_g | DEAmod.N_b) # Set of the union of column indicies -> so {1,2,3} - is dead code - unused

   # we do the same as above for the rows - set up RangeSets and a Set with the union of all row indicies
   DEAmod.M_g = RangeSet(1,MM_g)
   DEAmod.M_b = RangeSet(1,MM_b)
   DEAmod.M = Set(initialize = DEAmod.M_g | DEAmod.M_b)

   # variables in pyomo are things are numbers that the solve will fill in - what it picks.
   # It'll take later constraints and our objective, and find the "best" value

   # Two types: indexed (thats the first two, creates as many as there are in the collection)
   # Singe: empty parameter - just creates one.
   DEAmod.g = Var(DEAmod.N_g) # one for each good metric/col - a weight per metric
   DEAmod.b = Var(DEAmod.N_b) # one for each bad metric/col - a weight per metric
   DEAmod.v = Var() # a "baseline" - will define later.
   DEAmod.eps = Var() # efficiency score

   # Define "rules"
   # First, one for the objective.
   # Taken together, this fn defines the attirbute we want to optimize for
   # and the Objective set below, is saying - hey maximize this

   # Always return an expression - anything that can be evaluated to produce a value
   def obj_rule (DEAmod):
      return DEAmod.eps

   DEAmod.obj = Objective(rule = obj_rule, sense = maximize)

   ### ignoring for now ###
   #def obj_rule (DEAmod):
    #  return sum(a_g[iHat,i]*DEAmod.g[i] for i in DEAmod.N_g)- sum(a_b[iHat,j]*DEAmod.b[j] for j in DEAmod.N_b) - DEAmod.v
   #DEAmod.obj = Objective(rule = obj_rule, sense = maximize)
   
   # This time we're pairing a rule with a Constraint rather than an Objective
   # This calculates, for each row, the sum of each metric times its weight
   # We don't know the weight! That's for the solver to figure out - we added it as a variable, remember.
   # Also, the baseline... just keep in mind we want this to be as low as possible to get the higest possible EPS
   def con_eps_rule(DEAmod):
	   return DEAmod.eps + DEAmod.v - sum(a_g[iHat,i]*DEAmod.g[i] for i in DEAmod.N_g)+ sum(a_b[iHat,j]*DEAmod.b[j] for j in DEAmod.N_b)  == 1.0
   
   DEAmod.con_eps = Constraint(rule = con_eps_rule)

   # this is the core constraint, hence principal.
   # It takes one constraint per plan, because we set it up that way about with the indexed variable
   # So this is an "Indexed constraint" enforcing the good score of a plan minus the bad score of a plan is <= v - the baseline.
   # Now, the solver picks the baseline balancing the rules we're attaching.
   # i "walks across" the columns of a given plan's row and sums em, same for the bad and subtracts
   def con_principal_rule (DEAmod, m):
      return sum(a_g[m,i]*DEAmod.g[i] for i in DEAmod.N_g) - DEAmod.v - sum(a_b[m,j]*DEAmod.b[j] for j in DEAmod.N_b) <= 0.0
   DEAmod.con_principal = Constraint(DEAmod.M, rule=con_principal_rule)

   # now constaint that is just an expr - you can pass a short form.
   # we could have done it above
   # this is stating that the bar/baseline v cannot go below 1
   DEAmod.con_v = Constraint(expr = DEAmod.v >= 1.0)

   # These next rules ensure that we don't assign a weight of 0 to any given metrics,
   # and thus ignore a metric completely.
   # It assigns a floor, dynamically, depending on the eps it is also solving for and the row-metric value.
   # The floor is the minimum share. And for a given eps. We do it once for the bad and once for the good.

   def con_u_g_rule(DEAmod, ng):
	   return (S*a_g[iHat,ng])*DEAmod.g[ng] - DEAmod.eps >= 0.0 
   
   DEAmod.con_u_g = Constraint(DEAmod.N_g, rule = con_u_g_rule)
	
   def con_u_b_rule(DEAmod, nb):
      return (S*a_b[iHat,nb])*DEAmod.b[nb] - DEAmod.eps >= 0.0 
   
   DEAmod.con_u_b = Constraint(DEAmod.N_b, rule = con_u_b_rule) 
   
   # Create the concrete model and solve
   DEAinstance = DEAmod.create_instance() # builds the model

   # assign a solve that actually does the math
   # solver set to gurobi for 9 metrics, glpk for 3 and 6 metrics? Check this..Yes Gurobi licence expired
   Opt = SolverFactory("glpk")
   #Opt = SolverFactory("gurobi")

   # sovle and get a report back
   Soln = Opt.solve(DEAinstance)

   # this line is redundant, solve() above already loads
   DEAinstance.solutions.load_from(Soln)

   # return the a success tupe (score,0) or failure tupel (0,1)
   # however, this is broken ebcause we're saying if "optimal == optimal"
   if str(TerminationCondition.optimal) == "optimal":
      return value(DEAinstance.obj),0
   else:
      return 0,1


# and now the execution loop
# loop over each plan - 1 based to match the model

count = -1;
for iHat in range(1,MM+1):
   # score it (tuple without parens again)
	eff,failStat = calcEfficiency(a_g,a_b, MM_g,NN_g,MM_b, NN_b,MM, iHat)
   # if it failed print message satying which failed
	if failStat:
		count += 1
		#print "DMU "+str(iHat)+" did not solve"
		print("DMU "+str(iHat)+" did not solve")
	else:
      # otherwise increment the count for labeling and append results to csv
		count += 1
		print("count = ", count)
		print("DMU "+str(iHat)+" has efficiency "+str(eff))
		df_eff = df_eff.append({'Plan_Number_ID': count, 'DEA_Eff': eff},ignore_index=True)

df_eff.to_csv('df_eff_g_b_EPS_justinplussteve_nov142025.csv')





