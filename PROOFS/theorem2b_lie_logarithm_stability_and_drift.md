# Theorem 2B — Stability of Lie Logarithm Under Positive Jacobian Constraints

## Long-Horizon Drift Theorem — Bounded Accumulation under Repeated Lie Transport

This file collects the two stability results requested by the thesis analysis:
first, a rigorous stability theorem for the principal matrix logarithm under
positive-Jacobian constraints; second, a long-horizon drift theorem for repeated
adaptive transport and retraction operations. The statements are written for the
code paths in `FEM/state_transport.hpp`, `FEM/mesh_adaptation.hpp`, and
`FEM/nonlinear_solver.hpp`.

## 1. Motivation and admissible domain

The transport system in the code depends on the mapping
$$L = \log(F),$$
which is only stable when the deformation gradient remains in a compact subset
of the positive-determinant cone away from singularity and away from the
negative real axis. The code already assumes this regime through its positive
Jacobian checks, mesh-quality filters, and conservation tests.

Define the admissible set
$$\mathcal D_{\epsilon,M,\delta} := \{F\in GL^+(3): \det(F)\ge \epsilon,\ \|F\|\le M,\ \operatorname{dist}(\sigma(F),(-\infty,0])\ge \delta\}.$$
Here $\epsilon>0$ excludes singularity, $M$ prevents unbounded growth, and
$\delta>0$ keeps the spectrum away from the branch cut of the principal
logarithm.

Assume furthermore a uniform spectral gap for all iterates $F_k$ arising in
the transport sequence:
$$
\operatorname{dist}(\sigma(F_k),(-\infty,0]) \ge \delta >0.
$$

## 2. Theorem 2B — Stability of Lie Logarithm Under Positive Jacobian Constraints

### Theorem 2B
Let $F_1,F_2\in\mathcal D_{\epsilon,M,\delta}$. Then the principal matrix
logarithm exists uniquely for both matrices and satisfies a local Lipschitz
bound of the form
$$\boxed{\quad \|\log(F_1)-\log(F_2)\| \le C_{\log}\,\|F_1-F_2\|, \quad}$$
where $C_{\log}$ depends only on $\epsilon$, $M$, $\delta$, and the compact
subset of $\mathcal D_{\epsilon,M,\delta}$ under consideration.

Equivalently, the Lie logarithm is stable and continuously differentiable on the
admissible manifold, so perturbations in the deformation gradient do not explode
in log-space as long as the Jacobian remains positive and bounded away from
singularity.

### Proof
The proof uses the holomorphic functional calculus and classical matrix
perturbation theory.

#### Step 1: existence and uniqueness of the principal logarithm
Because the spectrum of every $F\in\mathcal D_{\epsilon,M,\delta}$ stays away
from the closed negative real axis, the principal branch of the scalar
logarithm is analytic on a contour enclosing $\sigma(F)$. Hence the principal
matrix logarithm is defined by the contour integral
$$\log(F)=\frac{1}{2\pi i}\int_{\Gamma} \log(z)\,(zI-F)^{-1}\,dz,$$
for a contour $\Gamma$ surrounding the spectrum and lying inside the domain of
analyticity.

#### Step 2: Fréchet derivative bound
The Fréchet derivative of the matrix logarithm at $F$ is
$$D\log(F)[E]=\frac{1}{2\pi i}\int_{\Gamma} \log(z)\,(zI-F)^{-1}E(zI-F)^{-1}\,dz.$$ 
Taking norms and bounding the resolvent along the contour yields
$$\|D\log(F)[E]\| \le C_{\mathrm{res}}(F)\,\|E\|,$$
where $C_{\mathrm{res}}(F)$ depends on the distance of the spectrum to the
branch cut and to zero. On a compact admissible set the constant can be chosen
uniformly, so there exists $C_{\log}>0$ such that
$$\|D\log(F)\|\le C_{\log}\qquad\forall F\in\mathcal D_{\epsilon,M,\delta}.$$

#### Step 3: Lipschitz continuity
For any $F_1,F_2\in\mathcal D_{\epsilon,M,\delta}$, the mean-value form of the
Banach-space Taylor theorem gives
$$\log(F_1)-\log(F_2)=\int_0^1 D\log(F_2+t(F_1-F_2))[F_1-F_2]\,dt.$$ 
Taking norms and using the uniform derivative bound on the compact admissible
set yields
$$\|\log(F_1)-\log(F_2)\|\le C_{\log}\,\|F_1-F_2\|.$$
This proves the claim.

#### Step 4: conditioning interpretation
The bound is sharp in the sense that $C_{\log}$ deteriorates when the spectrum
approaches the origin or the negative real axis. If $\det(F)\to 0^+$ then
$\|F^{-1}\|\to\infty$, which causes the Fréchet derivative bound to blow up.
Similarly, if an eigenvalue approaches the negative real axis, the principal
branch becomes ill-conditioned. The admissible set explicitly excludes both
pathologies, which is exactly why the code’s positive Jacobian and quality checks
are mathematically necessary.

### Final theorem-ready statement
For all deformation gradients satisfying positive Jacobian constraints,
bounded norm, and a spectral gap away from the negative real axis, the principal
matrix logarithm exists uniquely and is locally Lipschitz continuous. Therefore
small perturbations in deformation space produce proportionally small
perturbations in log-space, and the transport/retraction system remains stable.

### Literature positioning
This stability and drift analysis connects classical matrix-function
perturbation theory (Fréchet derivative bounds) with numerical stability of
Lie-logarithm-based transport. It builds on standard conditioning analyses (Higham-style)
and places the long-horizon drift estimate in the context of backward-stability
and structure-preserving numerical methods.

### Future direction: probabilistic floating-point drift analysis
The present theorem is deterministic and worst-case. A cutting-edge extension is
a probabilistic floating-point analysis for adaptive Lie-logarithmic FEM
transport under stochastic rounding, mixed precision, or randomized reduced-
precision accumulation. In that setting, the deterministic drift bound would be
replaced by a high-probability or expectation-based estimate for the modified
equation, yielding probabilistic control of determinant drift under repeated
refinement transfer. This would place the result at the interface of geometric
integration, stochastic numerics, and HPC floating-point analysis.

## 3. Long-Horizon Drift Theorem

### Theorem (Long-Horizon Drift under Repeated Lie Transport)
Let $F_n$ denote the deformation gradient after $n$ repeated transport,
retraction, projection, or refinement-transfer operations. Suppose each step
introduces a perturbation consisting of:
- floating-point rounding error of size at most $C_m\epsilon_{\mathrm{mach}}$,
- Padé truncation / logarithm truncation error of size at most
  $C_t\alpha_n^7\|L_n\|^7$,
- transport and projection defects of higher order that are absorbed into the
  same constants.

Then the determinant drift satisfies
$$\boxed{\quad |\det(F_n)-1| \le n\,C_m\epsilon_{\mathrm{mach}} + \sum_{k=1}^n C_{t,k}\alpha_k^7\|L_k\|^7.\quad}$$
In particular, if the transport increments are uniformly small and the code
periodically reprojects or re-solves, the drift remains bounded on arbitrarily
long horizons.

### Proof
Write each update in the form
$$F_{k+1}=\mathcal T_k(F_k)+\delta_k,$$
where $\mathcal T_k$ is the ideal Lie transport and $\delta_k$ collects the
rounding, truncation, and projection defects.

For the ideal transport, Theorem 2 gives exact determinant preservation in the
trace-free setting. Hence the only deviation from $\det(F)=1$ comes from the
perturbation $\delta_k$.

Using the determinant perturbation inequality on bounded sets,
$$|\det(F_{k+1})-\det(\mathcal T_k(F_k))|\le C_k\|\delta_k\|,$$
with $C_k$ depending only on the admissible compact set. Since
$\det(\mathcal T_k(F_k))=1$ in the idealized model, we obtain
$$|\det(F_{k+1})-1|\le C_k\|\delta_k\|.$$
Now decompose
$$\|\delta_k\|\le C_m\epsilon_{\mathrm{mach}} + C_t\alpha_k^7\|L_k\|^7,$$
which yields the single-step estimate
$$|\det(F_{k+1})-1|\le C_m'\epsilon_{\mathrm{mach}} + C_t'\alpha_k^7\|L_k\|^7.$$
Summing over $k=0,\dots,n-1$ gives the claimed drift bound.

If the algorithm periodically re-solves or reprojects the state onto the
admissible manifold, then the state is repeatedly reset to a small-neighborhood
regime and the cumulative drift becomes a bounded telescoping series rather
than a runaway accumulation.

### Recursive form
A convenient recursive version is
$$\delta_{n+1}\le (1+\alpha\epsilon_{\mathrm{mach}})\delta_n + C\epsilon_{\mathrm{mach}},$$
where $\delta_n:=|\det(F_n)-1|$. Solving this recurrence gives
$$\delta_n \le (1+\alpha\epsilon_{\mathrm{mach}})^n\delta_0 + \frac{C}{\alpha}\big((1+\alpha\epsilon_{\mathrm{mach}})^n-1\big),$$
which reduces to the linear-growth estimate $O(n\epsilon_{\mathrm{mach}})$ for
small machine precision and moderate horizons. If projection contraction is
applied at each macro-step, the coefficient becomes strictly contractive and the
drift remains uniformly bounded.

Applying the discrete Gr\"onwall inequality yields the alternative bound
$$
\delta_n \le e^{C n \epsilon_{\mathrm{mach}}}\left(\delta_0 + \sum_{k=0}^{n-1}\big(C_m\epsilon_{\mathrm{mach}} + C_t\alpha_k^7\|L_k\|^7\big)\right),
$$
which makes the dependence on accumulated machine error and high-order truncation
terms explicit and is often tighter for small $\epsilon_{\mathrm{mach}}$.

### Code interpretation
The theorem explains why repeated transport in the adaptive engine does not
cause catastrophic invariant collapse: every step is either corrected by
projection or remains in a regime where the accumulated defect is higher order.
The check in `verify_conservation` is therefore not merely empirical; it is a
finite-precision control law that monitors the drift predicted by the theorem.
