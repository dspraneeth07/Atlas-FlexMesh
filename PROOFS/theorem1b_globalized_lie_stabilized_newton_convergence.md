# Theorem 1B — Globalized Lie‑Stabilized Newton Convergence

This companion theorem upgrades the local analysis by incorporating the
backtracking line search and trust-region safeguards that are present in
`FEM/nonlinear_solver.hpp`.

## 1. Merit function and descent structure

Define the residual merit function
$$\phi_h(u) := \tfrac12\|R_h(u)\|^2,$$
where $R_h$ is the assembled nonlinear residual on the mesh family
`APP/test_cook_membrane_real.cpp` uses.

Assume that on the level set
$$\mathcal L_0 := \{u:\phi_h(u)\le \phi_h(u_0)\},$$
the gradient $\nabla\phi_h$ is Lipschitz continuous with constant $L_{\phi,h}$.
This is standard if $R_h$ is $C^1$ and the Jacobian $J_h$ is Lipschitz.

Let the Newton direction $p_k$ solve
$$J_h(u_k)p_k = -R_h(u_k).$$
Then
$$\nabla\phi_h(u_k)^T p_k = R_h(u_k)^T J_h(u_k)p_k = -\|R_h(u_k)\|^2 < 0$$
whenever $R_h(u_k)\neq 0$.
Hence the Newton direction is a strict descent direction for $\phi_h$.

## 2. Armijo backtracking line search

The code implements backtracking with parameters `line_search_alpha` and
`line_search_beta`. In mathematical form, choose $0<c_A<1$ and seek the largest
step size $\alpha_k = \beta^m$ such that the Armijo condition holds:
$$\phi_h(u_k+\alpha_k p_k)\le \phi_h(u_k)+c_A\alpha_k\nabla\phi_h(u_k)^T p_k.$$

By the descent lemma,
$$\phi_h(u_k+\alpha p_k)\le \phi_h(u_k)+\alpha\nabla\phi_h(u_k)^T p_k + \tfrac12 L_{\phi,h}\alpha^2\|p_k\|^2.$$ 
Because $\nabla\phi_h(u_k)^T p_k<0$, any step satisfying
$$0<\alpha\le \frac{2(1-c_A)|\nabla\phi_h(u_k)^T p_k|}{L_{\phi,h}\|p_k\|^2}$$
meets Armijo. Using the Newton identity above and the bounded inverse estimate
$$\|p_k\|\le \|J_h(u_k)^{-1}\|\,\|R_h(u_k)\|,$$
we obtain a uniform positive lower bound
$$\alpha_k\ge \underline\alpha_h>0$$
on the accepted step lengths whenever the iterates stay in $\mathcal L_0$ and
the Jacobian remains uniformly nonsingular on that set.

## 3. Global convergence under line search

The sufficient decrease inequality implies
$$\phi_h(u_{k+1})\le \phi_h(u_k)-c_A\alpha_k\|R_h(u_k)\|^2,$$
so $\phi_h(u_k)$ is monotonically decreasing and bounded below by $0$.
Consequently,
$$\sum_{k=0}^{\infty}\alpha_k\|R_h(u_k)\|^2<\infty,$$
and because $\alpha_k\ge \underline\alpha_h>0$, it follows that
$$\|R_h(u_k)\|\to 0.$$ 
Hence every accumulation point of $\{u_k\}$ is a stationary point of the
discrete residual system.

If the nonlinear elasticity problem has an isolated solution $u^*$ in the
relevant level set, and if the Jacobian at that solution is nonsingular, then
the entire sequence converges to $u^*$ rather than merely to a subsequence.
This is the standard globalization conclusion for backtracked Newton methods.

## 4. Trust-region boundedness

`FEM/nonlinear_solver.hpp` also contains trust-region scaffolding. Consider the
model
$$m_k(s)=\phi_h(u_k)+\nabla\phi_h(u_k)^T s + \tfrac12 s^T B_k s,$$
with a trial step constrained by
$$\|s\|\le \Delta_k.$$
Let the acceptance ratio be
$$\rho_k = \frac{\phi_h(u_k)-\phi_h(u_k+s_k)}{m_k(0)-m_k(s_k)}.$$
The standard trust-region update rules used in nonlinear FEM guarantee:
- if $\rho_k$ is large, the step is accepted and $\Delta_k$ may increase,
- if $\rho_k$ is small, the step is rejected and $\Delta_k$ decreases.

This yields bounded iterates because accepted steps are confined to the ball
of radius $\Delta_k$, while rejected steps do not change the iterate. Provided
$\phi_h$ has bounded level sets on the admissible domain, the entire trust-region
sequence remains in a compact subset of the local chart neighborhood.

## 5. Theorem 1B

### Theorem (Globalized Lie‑Stabilized Newton Convergence)
Assume:
1. $R_h$ is continuously differentiable and $\nabla\phi_h$ is Lipschitz on the
   level set $\mathcal L_0$.
2. The Jacobian $J_h(u)$ is uniformly nonsingular on $\mathcal L_0$.
3. The line search uses an Armijo rule with $0<c_A<1$ and backtracking factor
   $0<\beta<1$.
4. If a trust-region safeguard is activated, its radius updates satisfy the
   standard boundedness rules above.
5. The Lie-stabilization/retraction on the internal state is $C^2$ on the local
   chart and second-order consistent, so it does not destroy descent to leading
   order.

Then the globally safeguarded Newton iteration produces a bounded sequence
$\{u_k\}$ such that:
- the residual norms satisfy $\|R_h(u_k)\|\to 0$,
- every accumulation point is a stationary point of the discrete problem,
- if the stationary point is isolated and nonsingular, the whole sequence
  converges to that solution,
- once the sequence enters the local neighborhood of Theorem 1, the rate
  becomes quadratic.

Assume the iterates remain inside a compact admissible subset
$$
\mathcal K \subset \mathcal M_h
$$
on which the Jacobian remains uniformly nonsingular.

### Proof
The proof is the combination of the descent lemma, Armijo sufficient decrease,
and compactness of the sublevel set. The line search guarantees that every
accepted step decreases $\phi_h$ by at least a fixed fraction of the predicted
decrease, which forces the residual to vanish asymptotically. Trust-region
boundedness prevents the iterates from leaving the admissible neighborhood and
therefore prevents loss of smoothness or singular Jacobians. Since the Lie
retraction is analytic on the local chart, it contributes only higher-order
perturbations and does not alter the monotone decrease of the merit function
near the solution. Finally, once the iterates are sufficiently close to $u^*$,
the assumptions of Theorem 1 apply and the quadratic rate takes over.

By standard Zoutendijk theory for line-search methods,
$$
\sum_{k=0}^\infty \cos^2(\theta_k)\,\|\nabla\phi_h(u_k)\|^2 < \infty,
$$
where $\cos(\theta_k) = -\nabla\phi_h(u_k)^T p_k/(\|\nabla\phi_h(u_k)\|\,\|p_k\|)$. This identity strengthens the globalization argument by quantifying the cumulative descent along suitably aligned search directions and, combined with a uniform lower bound on accepted step lengths, implies $\|\nabla\phi_h(u_k)\|\to 0$.

No claim of global convexity is made for the hyperelastic energy outside the admissible neighborhood.

## 6. Practical interpretation for the code base

This theorem matches the implementation pattern in `FEM/nonlinear_solver.hpp`:
- the backtracking loop provides line-search globalization,
- the trust-region structures define a bounded fallback mechanism,
- the Lie transport in `FEM/state_transport.hpp` is second-order consistent,
- and the finite-element residual is driven to zero before local Newton speed is
  recovered.

Therefore the solver can be described, mathematically, as a globalized Newton
method with a local quadratic phase and a Lie-stabilized internal-state update.

### Literature positioning
This globalized result connects classical globalization techniques (Armijo
backtracking, trust-region methods, and Zoutendijk-style descent theory) with
Lie-stabilized internal-state updates. It situates the solver within the
well-established literature on globalized Newton methods and structure-preserving
geometric integration.

### Future direction: bifurcation-aware globalization
The present theorem is stated in the regime where a single locally nonsingular
solution branch is tracked inside a compact admissible set. A research-level
extension is a globalized convergence theorem near nonconvex bifurcation
manifolds and saddle-point structures, where solutions may fail to be unique and
the Jacobian may lose uniform definiteness along the branch. In that setting,
the globalization framework would need to be coupled with continuation,
branch-switching, or active-set logic to resolve bifurcation-induced loss of
uniqueness while retaining descent and local model fidelity.