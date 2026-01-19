## HyPhi Example Code

### Generating Small World Networks

Below, we demonstrate how to generate unweighted and weighted small world networks. We show example code for both the single network case and the "time varying" case in which the rewiring probability is varied.

```python

from Entropies import *
from GraphSimulations import *

# Pseudorandom number generation seed
# Needed for reproducibility
seed_val = 42

# Parameters for single small world network simulation
n = 1000    # Number of nodes in the network
k = 50      # Average node degree
p = 0.2     # Edge rewiring probability
ε = 1.0     # Unit spacing of nodes around the ring

# Generate a single unweighted small world network
# This is just the networkx implementation
# G is a networkx graph
G = nx.watts_strogatz_graph(n, k, p, seed=seed_val)

# Generate a single weighted small world network
# Gw is a networkx graph
Gw = genWeightedSW(n, k, p, ε, seed_val)

# Parameters for "time varying" small world networks
minpow = -4     # Exponent for lower bound of rewiring probability
maxpow = 0      # Exponent for upper bound of rewiring probability
trez = 1000     # Number of sample points for rewiring probability

# Generate a sequence of "time varying" unweighted small world networks
# Where the rewiring probability evolves from 10^minpow to 10^maxpow
# pt is the array of rewiring probabilities
# Gt is a list of networkx graphs at those rewiring probabilities
pt, Gt = genTVSW(n, k, trez, minpow, maxpow, seed_val)

# Generate a sequence of "time varying" unweighted small world networks
# Where the rewiring probability evolves from 10^minpow to 10^maxpow
# ptw is the array of rewiring probabilities
# Gtw is a list of networkx graphs at those rewiring probabilities
ptw, Gtw = genTVWeightedSW(n, k, ε, trez, minpow, maxpow, seed_val)
```

### Computing Graph Curvatures 

Below, we demonstrate the method for computing the edge Forman-Ricci curvatures of single networks and arrays of networks. The Ollivier-Ricci curvature case is more complicated and will be reserved for a separate tutorial.

```python

# Compute the Forman-Ricci curvature for a 
# single weighted network (same for unweighted)
# The "method_val" argument can be '1d' or 'augmented'
# FRC is a new network where edges have the property "formanCurvature" 
FRC = getFRC(Gw, method_val="1d")

# Extract the Forman-Ricci curvatures of the network into an array
curvatures = extractCurvatures(FRC, curvature="formanCurvature")

# Compute the Forman-Ricci curvature for an array 
# of weighted small world networks (same for unweighted)
# The "method_val" argument can be '1d' or 'augmented' 
# FRCt is a list of new networks where edges have the property "formanCurvature" 
FRCt = getFRCVec(Gtw, method_val="1d")

# Extract the Forman-Ricci curvatures of the networks into a list of arrays
curvatures_t = extractCurvaturesVec(FRC, curvature="formanCurvature")
```

### Kernel Density Estimation of Graph Curvature Distributions

We can compute nonparametric estimates of the graph curvature densities from the array of edge curvature values. This functionality is based heavily on the KDEpy package. 

```python

from DensityEstimation import *

# Parameters for the kernel density estimation
kernel_type = "gaussian"    # Gaussian kernel
bw = "ISJ"                  # Sheather-Jones algorithm to optimize bandwidth
norm = 2                    # Exponent for the vector norm 

# Fit the TreeKDE estimator  
# We use the TreeKDE because it is faster than naive
# But unlike the FFTKDE, we can evaluate at arbitrary points
f = TreeKDE(kernel=kernel_type, bw=bw, norm=norm).fit(curvatures)  

# Evaluate KDE at the original data points  
fvals = f.evaluate(curvatures) 
```

### Computing Entropies and Quantiles of Graph Curvature Distributions 

We can compute entropies and quantiles of the graph curvature distributions. There are many methods available for computing the entropy, of which the most robust is the Kozachenko-Leonenko nearest neighbor estimator.

```python

# Number of nearest neighbors to use for the Kozachenko-Leonenko entropy estimate
nn_val = 4

# Get the Kozachenko-Leonenko estimate of the entropy of the Forman-Ricci curvatures
# Single network case
H = getEntropyKozachenko(Gw, curvature="formanCurvature", num_nn=nn_val)

# Get the Kozachenko-Leonenko estimate of the entropy of the Forman-Ricci curvatures
# Array of multiple networks
# We first need to create a lambda function for the estimator we want to use
hKL = lambda X: getEntropyKozachenko(X, curvature="formanCurvature", num_nn=nn_val)
# Now we pass this lambda function to a parallelized entropy function 
Ht = vecEntropy(Gtw, estim=hKL)
```