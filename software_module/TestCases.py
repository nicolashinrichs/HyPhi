# ============= #
# Preliminaries # 
# ============= # 

from Entropies import *
from GraphSimulations import *
from scipy.stats import norm
import matplotlib.pyplot as plt

# Test the density estimation
# Generate a distribution and some Gaussian data
dist = norm(loc=0, scale=1)
data = dist.rvs(1000)

# Compute density estimates using Sheather Jones, compare to bw = 1
# bw = 1 is optimal since underlying distribution has std. dev. = 1
f_isj = fitKDE(data, bw="ISJ", method="tree")
x_isj, y_isj = f_isj()

f_truth = fitKDE(data, bw=1, method="tree")
x_truth, y_truth = f_truth()


# Replicate the Nature methods paper small world simulations
pt_nat, Gt_nat = genNatureSW()

FRCt_nat = getFRCVec(Gt_nat)

Ht_nat = vecEntropy(FRCt_nat, getEntropyKDEPlugin)