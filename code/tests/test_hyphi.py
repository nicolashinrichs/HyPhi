"""TODO: add docstring"""

# %% Import
from hyphi.modeling.entropies import vec_entropy, entropy_kde_plugin
from hyphi.modeling.density_estimation import fit_kde
from hyphi.modeling.graph_curvatures import compute_frc_vec
from hyphi.simulation.graph_simulations import gen_nature_sw
from scipy.stats import norm

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
pass

# %% Run tests >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# TODO: convert to proper pytest functionality

# Test the density estimation
# Generate a distribution and some Gaussian data
dist = norm(loc=0, scale=1)
data = dist.rvs(1000)

# Compute density estimates using Sheather Jones, compare to bw = 1
# bw = 1 is optimal since the underlying distribution has std. dev. = 1
f_isj = fit_kde(data, bw="ISJ", method="tree")
x_isj, y_isj = f_isj()

f_truth = fit_kde(data, bw=1, method="tree")
x_truth, y_truth = f_truth()


# Replicate the Nature methods paper small world simulations
pt_nat, Gt_nat = gen_nature_sw()

frc_t_nat = compute_frc_vec(Gt_nat)

ht_nat = vec_entropy(frc_t_nat, entropy_kde_plugin)

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
