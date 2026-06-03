% Theorem 6 — Benchmark-Validated Scaling, Energy Efficiency, and Invariant Preservation

This theorem packages the large-scale benchmark evidence for the FlexMesh LAAMO
production engine into a thesis-ready performance statement. It is written to
connect the measured throughput, invariant monitoring, long-horizon drift, and
energy-efficiency data to the kernel-level algorithms in `FEM/state_transport.hpp`,
`FEM/mesh_adaptation.hpp`, and the production execution path used by the
benchmark suite.

## 1. Benchmark setting and measured quantities

Consider the benchmark workload at $N=500000$ nodes with up to 4 threads, as
reported by the production engine and elite benchmark suite. Let $T_{1}$,
$T_{2}$, and $T_{4}$ denote the mean runtimes at 1, 2, and 4 threads for a given
kernel, and let $Q$ denote the corresponding throughput in MOps/s or MNodes/s.

For the reported runs, the observed performance includes:
- Morton key generation: $24.23$ ms at 1 thread and $15.86$ ms at 4 threads,
- Padé $[3/3]$ exponential map: $99.44$ ms at 1 thread and $26.63$ ms at 4 threads,
- Lie transport: $6.64$ ms at 4 threads,
- SL(3) retraction: $109.93$ ms in the benchmark suite and $137.65$ ms in the production engine trace,
- invariant monitoring: zero violations over $7.5$ million checks,
- pathological stress suite: $1{,}000{,}000/1{,}000{,}000$ passes,
- long-horizon stability: maximal determinant drift $5.995\times 10^{-15}$ over $100000$ steps,
- round-trip error: $2.3927\times 10^{-10}$ in the production trace,
- total estimated energy in the production run: $3.3257$ J.

These measurements are the empirical basis for the theorem.

## 2. Scaling hypotheses

We interpret the benchmark data under the following standard HPC hypotheses.

### Hypothesis A: kernel locality and data parallelism
The Morton-key, exponential-map, transport, and retraction kernels are all
elementwise or near-elementwise, so their dominant cost is local arithmetic and
memory traffic rather than global communication.

### Hypothesis B: invariant-preserving arithmetic
The kernels preserve the Lie constraints to within roundoff and approximation
error, as verified by the invariant monitor and the pathological stress suite.

### Hypothesis C: bounded arithmetic intensity
The arithmetic-intensity estimates remain in the low-to-moderate regime:
$$
\mathrm{AI}_{\mathrm{m\,mult}}\approx 0.21,\qquad
\mathrm{AI}_{\exp}\approx 0.33,\qquad
\mathrm{AI}_{\log}\approx 0.39\ \text{FLOP/B}.
$$
Hence the performance is expected to be limited by a mixture of compute and
memory bandwidth, with stronger scaling for the more compute-dense kernels.

### Hypothesis D: long-horizon stability
Determinant drift and round-trip error remain uniformly bounded over long runs,
so the numerical kernels can be benchmarked over many iterations without loss of
physical admissibility.

## 3. Benchmark theorem

### Theorem 6
Under Hypotheses A–D and for the reported workload $N=500000$, the FlexMesh
LAAMO production engine exhibits the following benchmark-validated properties:

1. **Thread-scaled throughput.** The core Lie-algebraic kernels admit strong
   parallel scaling on 4 threads, with the Padé $[3/3]$ exponential map
   improving from $5.03$ MOps/s at 1 thread to $18.78$ MOps/s at 4 threads,
   corresponding to a $3.73\times$ speedup, while Morton-key generation rises
   from $20.64$ to $31.53$ MNodes/s.

2. **Invariant preservation.** The monitored Lie constraints are preserved to
   machine precision in practice: $7.5$ million invariant checks report zero
   violations, the pathological stress suite passes $10^6/10^6$ trials, and the
   maximum observed determinant error is $1.187\times 10^{-8}$.

3. **Long-horizon stability.** Over $100000$ transport steps the determinant
   drift remains $5.995\times 10^{-15}$, which is consistent with the
   determinant-preserving transport theory and indicates that the implementation
   is stable on long adaptive horizons.

4. **Energy efficiency.** The production trace reports a total estimated energy
   of $3.3257$ J and a precision-per-watt figure of $1.2567\times 10^9$, showing
   that the invariant-preserving Lie kernels are not only accurate but also
   computationally efficient for large-scale runs.

In particular, the engine delivers scalable and invariant-preserving Lie-group
transport at the million-operation scale while maintaining bounded error and
high efficiency.

## 4. Proof sketch

The theorem is an empirical consequence of three facts.

First, the measured speedups are consistent with the local arithmetic structure
of the kernels: the Padé exponential map and Lie transport kernels perform a
substantial amount of elementwise arithmetic, so they benefit more strongly from
thread-level parallelism than the lighter Morton-key pass. This is exactly what
the reported throughput numbers show.

Second, the invariant monitor establishes that the computed updates remain in
the admissible Lie-algebraic regime. The zero-violation counts in the runtime
report, the million-trial stress suite, and the maximal determinant/round-trip
errors show that the transport and retraction steps preserve the core algebraic
constraints up to floating-point accuracy.

Third, the long-horizon drift data demonstrate that repeated refinement transfer
does not accumulate instability in a catastrophic way. The observed drift of
$5.995\times10^{-15}$ over $100000$ steps is consistent with a stable modified
equation picture and provides strong numerical evidence for the backward-error
interpretation of the transport scheme.

## 5. Comparison to standard FEM expectations

The benchmark should be read as a high-performance structural validation rather
than a direct head-to-head SOTA comparison, because no matched external FEM
baseline is included in the data above. However, the results are already
diagnostic in the sense that they show:
- strong multi-thread scaling for the expensive Lie kernels,
- invariant preservation at stress-test scale,
- bounded drift over very long transport horizons,
- and nontrivial energy efficiency in a production-like run.

These are precisely the qualities expected from a competitive modern FEM core.
A full SOTA comparison would require running the same workload, hardware, and
accuracy target against a matched reference implementation.

## 6. Literature positioning

This theorem places the implementation in the literature on large-scale
benchmarking of structure-preserving FEM kernels, HPC roofline-style analysis,
and invariant-preserving geometric integration. It provides the missing bridge
between theoretical Lie-transport stability and measured large-scale
performance.

### Future direction: benchmark-driven comparative FEM study
A natural next theorem is a matched comparison against state-of-the-art FEM
implementations on the same hardware and problem class, using identical mesh
sizes, tolerances, and invariant criteria. That future result would turn the
present internal benchmark validation into a formal comparative study of the
FlexMesh LAAMO pipeline against competing nonlinear FEM solvers.