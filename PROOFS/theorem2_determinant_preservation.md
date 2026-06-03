% Theorem 2 — Determinant Preservation under Lie‑Algebraic Transport and Retraction

This theorem is written for the transport operators in `FEM/state_transport.hpp`.
It makes the Padé(3,3) truncation order explicit, states the conditioning of the
matrix logarithm, and includes a long-horizon drift estimate for repeated
transport across multiple refinement levels.

## 1. Statement and geometric setting

Let $F_{\mathrm{parent}}\in GL^+(3)$ be a parent deformation gradient with
positive determinant. The code computes the transported child gradient by
calling `matrix_logarithm_sl3` and `matrix_exponential_sl3`, which correspond
to the idealized map
$$F_{\mathrm{child}} = \exp_{\mathrm{num}}\!\big(\alpha L\big),
\qquad L = \mathcal P_{\mathfrak{sl}(3)}(\log(F_{\mathrm{parent}})) + E_{\log},
\qquad 0\le\alpha\le1,$$
where $\mathcal P_{\mathfrak{sl}(3)}(A)=A-\tfrac13\operatorname{tr}(A)I$ is the
projection onto the trace-free Lie algebra $\mathfrak{sl}(3)$.

The admissible manifold is the positive-determinant set
$$\mathcal G_+ := GL^+(3) = \{F\in\mathbb R^{3\times3}: \det(F)>0\}.$$
Near any $F\in\mathcal G_+$ whose spectrum stays away from the negative real
axis and away from zero, the principal matrix logarithm is analytic.

## 2. Precise theorem

### Theorem 2
Assume the code operates in a compact subset of $\mathcal G_+$ where the
principal logarithm is defined and where the Jacobian determinant is bounded
away from zero. Suppose the logarithm approximation satisfies
$$\|E_{\log}\| \le \epsilon_{\log},$$
and the Padé(3,3) exponential approximation satisfies the order-seven remainder
estimate
$$\|E_{\exp}(A)\| \le C_7\|A\|^7$$
for all $\|A\|$ in the small-norm regime used by the guarded branch of
`matrix_exponential_sl3`.

Then there exists a computable constant $C>0$ such that
$$\boxed{\quad |\det(F_{\mathrm{child}})-1| \le C\big(\epsilon_{\log}+\alpha^7\|L\|^7\big).\quad}$$
In particular, for every $\epsilon>0$ there exist numerical tolerances such
that $|\det(F_{\mathrm{child}})-1|<\epsilon$.

## 3. Proof

### 3.1 Exact determinant identity for the matrix exponential
For any square matrix $A$,
$$\det(\exp(A)) = \exp(\operatorname{tr}(A)).\tag{1}$$
This follows from spectral mapping or from Jordan normal form.

### 3.2 Exact preservation in the ideal trace-free case
If $L_0=\mathcal P_{\mathfrak{sl}(3)}(\log(F_{\mathrm{parent}}))$ and
$\operatorname{tr}(L_0)=0$, then for any $\alpha\in[0,1]$,
$$\det(\exp(\alpha L_0)) = \exp(\alpha\operatorname{tr}(L_0)) = 1.$$
Hence the ideal Lie retraction preserves volume exactly.

### 3.3 Baker–Campbell–Hausdorff (BCH) structure
When successive Lie transports are composed, the Baker–Campbell–Hausdorff
expansion quantifies the noncommutativity of the increments. In particular
for sufficiently small $A,B$ one has
$$
\log(\exp(A)\exp(B)) = A+B+\tfrac12[A,B] + O(\|A\|\,\|B\|),
$$
where $[A,B]=AB-BA$ is the matrix commutator. This identity underpins the
geometric-integration viewpoint: the finite composition of small Lie updates
differs from their vector-space sum only by higher-order commutator terms.

### 3.4 Conditioning of the matrix logarithm and condition number
The principal logarithm is analytic on matrices whose spectrum avoids the
closed negative real axis. Define the logarithm condition number
$$
\kappa_{\log}(F) := \|D\log(F)\|.
$$
Its Fr\'echet derivative admits the standard contour representation, from which
one obtains a bound of the form
$$
\kappa_{\log}(F) \le \frac{\|F\|\,\|F^{-1}\|}{\operatorname{dist}(\sigma(F),(-\infty,0])}.
$$
Therefore the logarithm is well-conditioned only when the deformation gradient
remains separated from the branch cut and from singularity. As $\det(F)\to0^+$
or eigenvalues approach the negative real axis, $\kappa_{\log}(F)$ diverges.
The code's quality checks and guarded branches are precisely to keep
computations in a regime where $\kappa_{\log}(F)$ is moderate.

### 3.5 Numerical logarithm error
The implementation computes a Taylor approximation of $\log(I+E)$ and then
projects the result to $\mathfrak{sl}(3)$. Thus
$$L = L_0 + E_{\log},\qquad \operatorname{tr}(L_0)=0,\qquad \|E_{\log}\|\le\epsilon_{\log}.$$ 
Hence
$$|\operatorname{tr}(L)| = |\operatorname{tr}(E_{\log})| \le 3\|E_{\log}\| \le 3\epsilon_{\log}.\tag{2}$$

The attainable size of $\epsilon_{\log}$ in floating-point arithmetic depends on
the local backward/forward stability of the log routine and on
$\kappa_{\log}(F_{\mathrm{parent}})$: large $\kappa_{\log}$ amplifies
rounding and approximation errors, so in practice one enforces spectral
separation to keep $\epsilon_{\log}$ small.

### 3.6 Padé(3,3) remainder is order seven
The code uses the rational approximation
$$r_3(A)=\big(I+\tfrac12A+\tfrac1{12}A^2\big)\big(I-\tfrac12A+\tfrac1{12}A^2\big)^{-1}$$
after a small-norm safeguard. Since $r_3$ matches the exponential series
through degree six, the first nonzero local truncation term is degree seven.
Therefore there exists $C_7>0$ such that
$$\|\exp(A)-r_3(A)\|\le C_7\|A\|^7$$
for all sufficiently small $\|A\|$.

### 3.7 Determinant perturbation estimate
The computed child gradient satisfies
$$F_{\mathrm{child}} = \exp(\alpha L) + E_{\exp}(\alpha L),
\qquad \|E_{\exp}(\alpha L)\|\le C_7\alpha^7\|L\|^7.$$
Using (1) and the triangle inequality,
\begin{align*}
|\det(F_{\mathrm{child}})-1|
&\le |\det(\exp(\alpha L))-1| + |\det(\exp(\alpha L)+E_{\exp})-\det(\exp(\alpha L))| \\
&= |\exp(\operatorname{tr}(\alpha L))-1| + |\det(\exp(\alpha L)+E_{\exp})-\det(\exp(\alpha L))|.
\end{align*}

For the first term, using $|e^x-1|\le |x|e^{|x|}$ and (2),
\begin{align*}
|\exp(\operatorname{tr}(\alpha L))-1|
&\le |\operatorname{tr}(\alpha L)|e^{|\operatorname{tr}(\alpha L)|} \\
&\le 3\alpha\epsilon_{\log}e^{3\alpha\epsilon_{\log}}.
\end{align*}

For the second term, the determinant is multilinear in the columns and hence
Lipschitz on bounded sets. Thus there exists a constant $C_{\det}(\alpha,L)$
such that
$$|\det(\exp(\alpha L)+E_{\exp})-\det(\exp(\alpha L))|\le C_{\det}(\alpha,L)\,\|E_{\exp}(\alpha L)\|.$$
Consequently
\begin{align*}
|\det(F_{\mathrm{child}})-1|
&\le 3\alpha\epsilon_{\log}e^{3\alpha\epsilon_{\log}} + C_{\det}(\alpha,L)C_7\alpha^7\|L\|^7 \\
&\le C\big(\epsilon_{\log}+\alpha^7\|L\|^7\big)
\end{align*}
for a computable constant $C>0$. This proves the theorem.

## 4. Long-horizon drift theorem

### Theorem (Long-Horizon Drift under Repeated Transport)
Let $F_n$ be the deformation gradient after $n$ repeated transports or
refinement transfers. Assume each step incurs machine error bounded by
$O(\epsilon_{\mathrm{mach}})$ and a higher-order transport truncation bounded
by $O(\alpha^7\|L\|^7)$. Then
$$|\det(F_n)-1| \le n\,C_{\mathrm{mach}}\epsilon_{\mathrm{mach}} + \sum_{k=1}^n C_k\alpha_k^7\|L_k\|^7.$$
In particular, if the transport increments are uniformly small and
relinearization or re-solve is performed periodically, the drift remains
bounded on long horizons.

### Proof
At step $k$, write
$$F_{k+1}=\mathcal T_k(F_k)+\delta_k,$$
where $\mathcal T_k$ is the ideal transport and $\delta_k$ is the rounding and
Padé truncation perturbation. By the single-step theorem,
$$|\det(F_{k+1})-1|\le |\det(\mathcal T_k(F_k))-1| + C_k\|\delta_k\|.$$
The ideal transport preserves volume, so the first term is zero. Summing over
the sequence and using the triangle inequality yields the stated bound.

## 5. Consequences for the code

The code’s `verify_conservation` check enforces the practical corollary
$$|\det(F)-1|<10^{-8}$$
on every transported state. The theorem above explains why: the logarithm is
only used in a stable region of $GL^+(3)$, the trace is projected out exactly
in exact arithmetic, and the numerical remainder is seventh order in the small
transport increment.

### Literature positioning
The determinant-preserving transport relates to geometric numerical
integration and structure-preserving Lie-group methods. It connects the code's
practical safeguards to the broader theory of Lie-group integrators and
structure-preserving approximations in numerical analysis.

### Future direction: modified equation and backward error analysis
The Lie transport can also be interpreted through the lens of backward error
analysis: the discrete transport step is viewed as the exact flow of a nearby
modified equation whose vector field differs from the continuum transport law by
high-order commutator and truncation terms. In that interpretation, the BCH
terms and Padé remainder define the leading modified-equation corrections,
placing the determinant-preserving update in the standard framework used by
Hairer, Lubich, and the geometric integration literature. A fully developed
theorem would identify the modified transport equation whose exact solution is
shadowed by the numerical Lie update over long horizons.

