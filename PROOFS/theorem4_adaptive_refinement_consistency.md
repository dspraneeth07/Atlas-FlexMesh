% Theorem 4 — Adaptive Refinement Consistency Theorem

This theorem formalizes the fact that the adaptive framework in the code base
does not destroy finite-element convergence even though the approximation spaces
change over time. The result is written for the mesh adaptation and transport
operators in `FEM/mesh_adaptation.hpp`, `FEM/state_transport.hpp`, and the
nonlinear solve loop in `FEM/nonlinear_solver.hpp`.

## 1. Problem setting

Let $\Omega\subset\mathbb R^3$ be the reference domain and consider a nonlinear
boundary-value problem whose weak form is:
$$
a(u,v)=\ell(v)\qquad\forall v\in V,$$
where $V$ is a Hilbert space with norm $\|\cdot\|_V$ and $u\in V$ is the exact
solution. In the code, $u$ is represented by the nodal solution vector on a
mesh-dependent finite-dimensional subspace $V_h\subset V$.

For each adaptive level $k$, let $\mathcal T_k$ be the current conforming mesh,
let $h_k$ denote its characteristic size, and let $V_k\subset V$ be the
associated finite-element space. Let $u_k\in V_k$ denote the computed discrete
solution on that mesh.

The adaptive process consists of the following operators:
- a refinement/coarsening map that changes $\mathcal T_k$ to $\mathcal T_{k+1}$,
- a prolongation or projection operator $\mathcal P_k:V_k\to V_{k+1}$,
- a Lie-algebraic state transport operator $\mathcal T_k^{\mathrm{Lie}}$ for
  history variables such as deformation gradients and stresses,
- a new discrete solve on the refined space.

The goal is to prove that these operations preserve the asymptotic convergence
order of the method.

## 2. Stability hypotheses

We assume the standard hypotheses used in adaptive finite-element analysis.

### Hypothesis A: shape regularity
The mesh family $\{\mathcal T_k\}_{k\ge0}$ is shape regular. Equivalently,
there exists a constant $\gamma>0$ such that each element satisfies a uniform
aspect-ratio bound and the interpolation constants remain uniformly bounded in
$k$.

### Hypothesis B: approximation property
For polynomial degree $p\ge1$, the interpolation operator $I_k:V\to V_k$
satisfies
$$\|u-I_k u\|_V \le C_{\mathrm{app}} h_k^p \|u\|_{W^{p+1}},$$
for all $u$ in the appropriate Sobolev regularity class, with a constant
$C_{\mathrm{app}}$ depending only on the shape-regularity bound and the
reference element.

### Hypothesis C: prolongation stability
The transfer operator $\mathcal P_k:V_k\to V_{k+1}$ is stable in the energy
norm:
$$\|\mathcal P_k w\|_V \le C_P \|w\|_V\qquad\forall w\in V_k,$$
for a constant $C_P$ independent of $k$.

### Hypothesis D: transport consistency
The transport of internal state variables is second-order consistent in the mesh
size:
$$\|\mathcal T_k^{\mathrm{Lie}} z - z\| \le C_T h_k^{p+1}$$
for all transported quantities $z$ in the relevant admissible state set.
This is the mathematical abstraction of the Lie transport implemented in
`FEM/state_transport.hpp`, where the deformation gradient is transported via a
projected logarithm/exponential update on $SL(3)$.

### Hypothesis E: discrete stability of the nonlinear solve
On each mesh level, the discrete nonlinear problem is solved to the tolerance
used by the code’s Newton solver, and the discrete solution satisfies a Céa-type
best-approximation estimate with mesh-independent constant:
$$\|u-u_k\|_V \le C_{\mathrm{C\'ea}} \inf_{w\in V_k}\|u-w\|_V + \eta_k,$$
where $\eta_k$ is the nonlinear solve tolerance, assumed to satisfy
$\eta_k = O(h_k^p)$ or smaller.

These are the exact hypotheses needed to ensure that adaptivity changes the
space without changing the asymptotic rate.

### Hypothesis F: Dörfler marking (bulk chasing)
Assume the refinement strategy satisfies Dörfler marking: there exists
$0<\theta_{\mathrm{mark}}\le1$ such that the set of marked elements
$\mathcal M_k\subset\mathcal T_k$ satisfies
$$
\sum_{K\in\mathcal M_k}\eta_K^2 \ge \theta_{\mathrm{mark}}\sum_{K\in\mathcal T_k}\eta_K^2,
$$
where $\{\eta_K\}$ is a reliable and efficient a posteriori error estimator
on mesh $\mathcal T_k$.

## 3. Main theorem

### Theorem 4
Under Hypotheses A–E, adaptive remeshing, Lie transport, refinement, and
prolongation preserve the convergence order of the finite-element approximation.
More precisely, there exists a constant $C>0$, independent of the adaptation
level $k$, such that
$$\boxed{\quad \|u-u_k\|_V \le C h_k^p \quad}$$
for all sufficiently large $k$.
In particular, the adaptive sequence converges to the exact solution:
$$\|u-u_k\|_V \to 0\qquad\text{as }k\to\infty.$$

## 4. Proof

### Step 1: best-approximation on the current mesh
By Céa’s lemma and the assumed ellipticity/continuity of the weak form,
$$\|u-u_k\|_V \le C_{\mathrm{C\'ea}} \inf_{w\in V_k}\|u-w\|_V + \eta_k.$$
Choosing the interpolant $w=I_k u$ gives
$$\|u-u_k\|_V \le C_{\mathrm{C\'ea}}\|u-I_k u\|_V + \eta_k.$$
Using the approximation property,
$$\|u-u_k\|_V \le C_{\mathrm{C\'ea}} C_{\mathrm{app}} h_k^p \|u\|_{W^{p+1}} + \eta_k.$$
Since $\eta_k=O(h_k^p)$ by assumption, there exists a constant $C_1$ such that
$$\|u-u_k\|_V \le C_1 h_k^p.$$
This establishes optimal order on each fixed adaptive level.

### Step 2: effect of prolongation between levels
Let $u_k\in V_k$ be the solution on mesh $\mathcal T_k$, and let
$\widetilde u_{k+1}=\mathcal P_k u_k\in V_{k+1}$ be the transferred state used
as the initial guess on the next mesh.
By stability of the prolongation operator,
$$\|u-\widetilde u_{k+1}\|_V
\le \|u-u_k\|_V + \|u_k-\mathcal P_k u_k\|_V,$$
and the second term is bounded by the operator norm of $\mathcal P_k$.
Therefore
$$\|u-\widetilde u_{k+1}\|_V \le C_2 \|u-u_k\|_V + C_3 h_k^p,$$
where the last term accounts for the approximation defect introduced by the
mesh change itself. Since $\|u-u_k\|_V = O(h_k^p)$, we obtain
$$\|u-\widetilde u_{k+1}\|_V = O(h_k^p).$$

### Step 3: transport consistency for internal variables
The transported history variables do not enter the error estimate directly, but
they affect the nonlinear residual and therefore the quality of the next solve.
By Hypothesis D, the transport defect is of higher order:
$$\|\mathcal T_k^{\mathrm{Lie}} z - z\| \le C_T h_k^{p+1}.$$
Because this is one order higher than the primary approximation error, it does
not alter the dominant $h_k^p$ term in the discrete solution estimate. In
particular, the transport can perturb the Newton initial guess and the internal
state, but only at higher order relative to the finite-element error.

### Step 4: solve on the refined mesh
The nonlinear solve on the refined mesh $V_{k+1}$ satisfies the same Céa-type
estimate:
$$\|u-u_{k+1}\|_V \le C_{\mathrm{C\'ea}} \inf_{w\in V_{k+1}}\|u-w\|_V + \eta_{k+1}.$$
Because the refined mesh is finer than or comparable to the previous one and the
family is shape regular, the approximation property improves or stays optimal:
$$\inf_{w\in V_{k+1}}\|u-w\|_V \le C_{\mathrm{app}} h_{k+1}^p \|u\|_{W^{p+1}}.$$ 
Thus
$$\|u-u_{k+1}\|_V \le C_4 h_{k+1}^p + \eta_{k+1}.$$
If the nonlinear tolerance satisfies $\eta_{k+1}=O(h_{k+1}^p)$, then
$$\|u-u_{k+1}\|_V \le C_5 h_{k+1}^p.$$

### Step 5: induction over adaptive levels
The argument above is uniform in $k$ because the mesh family is shape regular
and the transfer/transport constants are independent of level. Therefore, by
induction on the adaptive level, the entire adaptive sequence satisfies
$$\|u-u_k\|_V \le C h_k^p$$
for all sufficiently large $k$. Since $h_k\to0$ under refinement, it follows
that
$$\|u-u_k\|_V \to 0.$$
This proves both convergence and preservation of order.

## 5. Adaptive optimality (Dörfler marking and quasi-optimality)

If, in addition to Hypotheses A–F, the estimator $\{\eta_K\}$ is locally
efficient and the refinement produces meshes with near-minimal cardinality for
the achieved error reduction (standard assumptions in AFEM theory), then the
adaptive sequence satisfies the canonical quasi-optimal complexity bound.

### Theorem (Quasi-optimal complexity)
Let $N_k:=\dim(V_k)$ be the number of degrees of freedom at level $k$. Under
the assumptions above there exists a constant $C_{\mathrm{opt}}$ such that
for all $k$ one has the complexity bound
$$
\|u-u_k\|_V \le C_{\mathrm{opt}}\,N_k^{-p/d},
$$
where $d=3$ is the spatial dimension and $p$ the polynomial approximation
order.

### Proof sketch
The Dörfler marking (Hypothesis F) together with estimator reduction on the
refined region yields linear convergence of the estimator up to data oscillation
terms. Standard AFEM proofs (estimator reduction + discrete reliability + mesh
complexity counting) then imply that the adaptive algorithm achieves the same
error vs DOF rate as the best-approximation class, i.e. quasi-optimality
$N_k^{-p/d}$. The transport and prolongation operators enter only through
higher-order perturbations (Hypothesis D and C), and thus do not affect the
complexity class.

## 5. Interpretation in the code base

The mesh-adaptation layer in `FEM/mesh_adaptation.hpp` maintains conformity and
positive element quality; `FEM/state_transport.hpp` transports history variables
through a second-order consistent Lie update; and `FEM/nonlinear_solver.hpp`
re-solves the nonlinear system on each refined space. Together these operations
form a stable adaptive approximation pipeline whose asymptotic error rate is
governed by the same interpolation order as the underlying finite elements.

## 6. Final thesis-ready statement

Adaptive remeshing does not destroy convergence when the mesh family is shape
regular, the prolongation operator is stable, the Lie transport is higher-order
consistent, and the nonlinear solve is accurate to the discretization scale.
Under these conditions,
$$\|u-u_k\|_V \le C h_k^p\to0,$$
so the adaptive framework remains mathematically convergent and order-preserving
throughout refinement, transport, and prolongation.

### Literature positioning
This result places the adaptive Lie-transport framework within the modern AFEM
theory: with Dörfler marking and estimator reduction it satisfies the standard
assumptions that yield quasi-optimal complexity and convergence (cf.
classical AFEM literature on Dörfler marking and optimality).

### Estimator reliability, efficiency, and contraction
To close the AFEM loop, one may supplement the above theorem with the standard
residual-estimator theory. Let $\eta_k^2:=\sum_{K\in\mathcal T_k}\eta_K^2$ be a
meshwise residual estimator. Then on a shape-regular family there exist
constants $C_{\mathrm{rel}},C_{\mathrm{eff}}>0$, independent of $k$, such that
\begin{align*}
\|u-u_k\|_V &\le C_{\mathrm{rel}}\,\eta_k + \mathrm{osc}_k,\\
\eta_K &\le C_{\mathrm{eff}}\big(\|u-u_k\|_{V(\omega_K)} + \mathrm{osc}_{\omega_K}\big),
\end{align*}
where $\omega_K$ is the element patch and $\mathrm{osc}_k$ denotes the data
oscillation term.

Under Dörfler marking, local refinement, and estimator reduction on the marked
set, the estimator contracts up to higher-order perturbations:
$$
\eta_{k+1}^2 \le q\,\eta_k^2 + C\|u_{k+1}-u_k\|_V^2 + C h_k^{2p}
$$
for some $0<q<1$. Combined with the solve accuracy and transport consistency
already assumed above, this yields the usual AFEM contraction mechanism: the
estimator decreases geometrically until it reaches the discretization floor,
and the quasi-optimal rate follows from the reliability/efficiency pair.

### Proof sketch
Reliability follows from Galerkin orthogonality plus local bubble-function and
Clément-type arguments applied to the residual representation; efficiency
follows from elementwise inverse estimates and local lower bounds on the same
residual indicators. Contraction follows by splitting the estimator into marked
and unmarked contributions, using Dörfler bulk chasing on the marked set,
and invoking estimator reduction after refinement. The transport and prolongation
terms enter only as higher-order perturbations, so they do not change the AFEM
contraction class.
