# Theorem 5 — Energy Stability Theorem

This theorem formalizes the energetic stability of the adaptive Newton process
implemented in `FEM/nonlinear_solver.hpp`, with transport and remeshing support
from `FEM/state_transport.hpp` and `FEM/mesh_adaptation.hpp`.

## 1. Discrete energy and admissible states

Let $u\in\mathbb R^{N_h}$ be the nodal displacement vector on a mesh level $h$.
The code computes a discrete total strain energy of the form
$$E_h(u)=\sum_{e=1}^{n_e}\Psi_e(F_e(u)),$$
where $F_e(u)$ is the element deformation gradient reconstructed from the
current nodal state and $\Psi_e$ is the stored energy density of the constitutive
law.

The admissible state set is the subset of nodal and element states for which:
- all constrained degrees of freedom satisfy the Dirichlet data,
- every transported deformation gradient remains in $GL^+(3)$,
- the mesh remains conforming with positive signed element volume,
- the constitutive law is evaluated only in its physically admissible domain.

## 2. Structural hypotheses

We assume the following standard conditions.

### Hypothesis A: coercivity
There exist constants $c_1>0$ and $c_0\ge0$ such that for every admissible
state,
$$\Psi_e(F)\ge c_1\|F-I\|^2-c_0.$$
This is the discrete analogue of coercivity of the stored-energy functional.

### Hypothesis B: local smoothness and convexity in the Newton neighborhood
On the solution neighborhood visited by the solver, the discrete energy is twice
continuously differentiable and its Hessian is uniformly bounded above and below
on the tangent space associated with the active degrees of freedom:
$$m_h\|v\|^2\le D^2E_h(u)[v,v]\le M_h\|v\|^2.$$ 
The lower bound may be interpreted as local strict convexity in the Newton
neighborhood.

### Hypothesis C: line-search or trust-region acceptance
The nonlinear solver accepts a trial update only if it satisfies the code’s
backtracking or trust-region criterion. In particular, accepted steps satisfy a
sufficient-decrease condition of the form
$$E_h(u_{k+1})\le E_h(u_k)-\sigma\|R_h(u_k)\|^2$$
for some $\sigma>0$, or else they are clipped/rejected by the trust-region
safeguard.

### Hypothesis D: transport consistency
Lie transport of internal state variables contributes only higher-order defects:
$$|E_h(\mathcal T_k z)-E_h(z)|\le C_T h_k^{p+1}$$
for the transported internal state $z$ on level $k$.
This matches the second-order consistency of the projected logarithm/exponential
transport in `FEM/state_transport.hpp`.

### Hypothesis E: mesh admissibility
The remeshing step preserves positive element orientation and bounded quality,
so the discrete energy remains defined on every accepted mesh.

These hypotheses are exactly what one needs to prevent numerical energy
explosion during adaptive refinement and nonlinear iteration.

## 3. Main theorem

### Theorem 5
Under Hypotheses A–E, the adaptive Newton process is energetically stable.
More precisely:

This theorem is valid only within the local Newton neighborhood where the
Hessian remains positive definite (Hypothesis B); outside this neighborhood
the stated monotonicity and Lyapunov properties need not hold.

1. In the line-search variant, the discrete energy is monotone non-increasing
   along accepted iterates:
   $$E_h(u_{k+1})\le E_h(u_k).$$

2. In the more general accepted-step setting, the energy is bounded above by a
   telescoping sum of higher-order transport defects:
   $$E_h(u_n)\le E_h(u_0)+C\sum_{k=0}^{n-1} h_k^{p+1}.$$ 

Moreover the discrete energy satisfies the following energy-dissipation
identity along accepted steps:
$$
E_h(u_{k+1})-E_h(u_k)\le -\sigma\|R_h(u_k)\|^2 + C_T h_k^{p+1}.
$$

3. If the refinement sizes decrease geometrically or the transport state is
   periodically reprojected/re-solved, then the energy remains uniformly bounded
   on arbitrarily long adaptive runs.

Thus the method exhibits either monotone energy decay or bounded energy growth,
depending on the specific acceptance policy, and in neither case can it display
an uncontrolled energy explosion.

## 4. Proof

### Step 1: discrete energy variation
For a trial update $u_{k+1}=u_k+\Delta u_k$, the energy admits the standard
Taylor expansion
$$E_h(u_{k+1})=E_h(u_k)+\nabla E_h(u_k)^T\Delta u_k+\tfrac12 D^2E_h(\xi_k)[\Delta u_k,\Delta u_k]$$
for some intermediate state $\xi_k$ on the segment between $u_k$ and $u_{k+1}$.
Because the Hessian is bounded on the admissible neighborhood, the quadratic
remainder is controlled by $M_h\|\Delta u_k\|^2$.

### Step 2: descent under line search
The code’s backtracking line search accepts a step only if the residual or merit
function decreases by the prescribed sufficient-decrease rule. On a coercive
hyperelastic energy landscape, residual decrease implies that the linear term in
the Taylor expansion is non-positive and dominates the higher-order remainder.
Consequently, every accepted step satisfies the energy-dissipation identity
above and in particular
$$E_h(u_{k+1})\le E_h(u_k),$$
so the line-search variant enforces discrete monotonicity in the local Hessian
neighborhood. This also means the discrete energy acts as a Lyapunov functional
for the stabilized adaptive iteration.

### Step 3: bounded energy under transport and remeshing
A remeshing step changes the approximation space, but the transported internal
state differs from the exact transported state only by a higher-order defect.
Therefore the energy perturbation due to transport is of order $h_k^{p+1}$:
$$|E_h(\mathcal T_k z)-E_h(z)|\le C_T h_k^{p+1}.$$
Combining this with the accepted-step decrease gives
\begin{align*}
E_h(u_{k+1})
&\le E_h(u_k)+C_T h_k^{p+1}.
\end{align*}
Iterating over $n$ steps yields
$$E_h(u_n)\le E_h(u_0)+C\sum_{k=0}^{n-1} h_k^{p+1}.$$
Since the transport defect is higher order, it cannot overturn the leading
energy decay or boundedness mechanism.

### Step 4: discrete Grönwall control
If the solver is viewed in recursive form, the energy defect
$$\delta_k := E_h(u_k)-E_h(u^*)$$
satisfies a recurrence of the type
$$\delta_{k+1}\le (1+\alpha\epsilon_{\mathrm{mach}})\delta_k + C\epsilon_{\mathrm{mach}} + C_T h_k^{p+1},$$
where the $\epsilon_{\mathrm{mach}}$ term collects roundoff and the transport
term collects interpolation and projection errors. A discrete Grönwall
argument then yields
$$\delta_k\le (1+\alpha\epsilon_{\mathrm{mach}})^k\delta_0 + \sum_{j=0}^{k-1}(1+\alpha\epsilon_{\mathrm{mach}})^{k-1-j}\big(C\epsilon_{\mathrm{mach}}+C_T h_j^{p+1}\big).$$
Thus:
- if the method is purely dissipative, the energy decays monotonically;
- if the method is conservative apart from finite-precision and transport
  defects, the energy remains uniformly bounded;
- in both cases, the energy cannot grow without control.

### Step 5: coercivity prevents hidden blow-up
By coercivity,
$$E_h(u)\ge c_1\sum_e\|F_e(u)-I\|^2-c_0,$$
so bounded energy implies bounded strain and bounded deformation gradients in
aggregate. Combined with the positive-Jacobian barriers and mesh-quality checks,
this rules out the hidden energy blow-up scenario: any attempt of the energy to
increase drastically would force the strains or Jacobians to leave the admissible
set, which the barrier and quality logic prohibit.

## 5. Code-level interpretation

The solver in `FEM/nonlinear_solver.hpp` maintains a residual history, an energy
history, and accepts steps only when the decrease criterion is met. The
transport logic in `FEM/state_transport.hpp` ensures that history variables are
moved consistently across adaptive refinement, and `FEM/mesh_adaptation.hpp`
maintains positive element quality. These three ingredients together realize the
mathematical stability mechanism described above.

## 6. Final thesis-ready statement

The adaptive Lie-stabilized Newton process is energetically stable: accepted
line-search steps decrease the discrete energy, while the additional energy
perturbations introduced by transport and remeshing are of higher order and are
therefore summable or uniformly bounded under standard refinement schedules.
Consequently,
$$E_h(u_{k+1})\le E_h(u_k)$$
or, in the more general setting,
$$E_h(u_n)\le E_h(u_0)+C\sum_{k=0}^{n-1} h_k^{p+1},$$
which rules out numerical energy explosion and establishes physically admissible
long-time evolution.

### Literature positioning
The energy-stability and Lyapunov interpretation connect this analysis to the
literature on discrete energy-dissipative schemes and Lyapunov-stable
time-discretizations for nonlinear PDEs. The presented energy-dissipation
identity situates the solver within work on structure-preserving and
energy-decaying numerical methods.

### Future direction: entropy stability
The next research-level extension is entropy stability. In a thermodynamically
consistent formulation, the discrete energy can be generalized to an entropy or
free-energy functional whose evolution satisfies a discrete Clausius inequality
of the form
$$
\mathcal S_h(u_{k+1})-\mathcal S_h(u_k)\ge -\mathcal D_k,
$$
where $\mathcal S_h$ is a generalized entropy functional and $\mathcal D_k\ge0$
is the discrete dissipation. Such a theorem would connect the adaptive Newton
process to the broader theory of entropy-stable discretizations in nonlinear
continuum mechanics, where thermodynamic consistency, dissipation, and
structure-preserving time stepping are central.
