# Geometric Successive Refinement for Matryoshka Representations

## One-sentence thesis

Matryoshka prefixes should not independently imitate the same full embedding; each newly added coordinate band should recover the **next residual component of a corpus-level spectral geometry**.

## 1. Motivation

Let an encoder produce an embedding $z=f_\theta(x)\in\mathbb R^D$, and let

$$
0=d_0<d_1<\cdots<d_K=D
$$

be the supported Matryoshka dimensions. Standard MRL applies a task loss to every prefix $z_{1:d_k}$. This makes every requested prefix useful, but it does not specify what the newly introduced coordinates

$$
z_{d_{k-1}+1:d_k}
$$

should add beyond the previous prefix.

Whole-prefix alignment does not resolve this allocation problem when its similarity is scale invariant, and redundancy penalties only state what a new band should avoid. We instead assign every band a positive target: the spectral geometry that becomes available at its additional dimensional budget.

We call the resulting protocol **Geometric Successive Refinement (GSR)**.

## 2. Geometry branch

### 2.1 Full-normalized representations

For the geometry branch, normalize only once at the full dimension:

$$
q_\theta(x)=\frac{f_\theta(x)}{\max(\|f_\theta(x)\|_2,\varepsilon)}.
$$

This choice has two useful properties.

First, the full-dimensional geometry is angular:

$$
\|q_n-q_m\|_2^2=2-2\cos(z_n,z_m).
$$

Second, coordinate-band distances remain exactly additive because every band shares the same full-vector denominator. Prefix normalization used by the task loss is unaffected:

$$
\operatorname{norm}(q_{1:d})=\operatorname{norm}(z_{1:d}).
$$

Thus the task branch retains standard cosine-based MRL, while the geometry branch obtains an additive decomposition aligned with the full angular representation.

### 2.2 A corpus-level spectral teacher

At outer iteration $t$, freeze a teacher snapshot $\bar\theta_t$ and encode the training corpus $\mathcal D=\{x_n\}_{n=1}^N$ deterministically:

$$
Q^T=[q_{\bar\theta_t}(x_1),\ldots,q_{\bar\theta_t}(x_N)]^\top\in\mathbb R^{N\times D}.
$$

Let

$$
X^T=C_NQ^T,
\qquad
C_N=I-\frac1N\mathbf 1\mathbf 1^\top,
$$

and define the corpus covariance

$$
\Sigma_T=\frac1N(X^T)^\top X^T
=V\Lambda V^\top,
\qquad
\lambda_1\ge\cdots\ge\lambda_D\ge0.
$$

For band $I_k=(d_{k-1},d_k]$, write

$$
V_k=V[:,d_{k-1}+1:d_k],
\qquad
P_k=V_kV_k^\top.
$$

The corresponding global teacher shell is

$$
R_k=X^TP_k(X^T)^\top.
$$

The projectors are mutually orthogonal and resolve the identity:

$$
P_kP_j=0\;(k\ne j),
\qquad
\sum_{k=1}^K P_k=I.
$$

Consequently,

$$
R_k\succeq0,
\qquad
\operatorname{rank}(R_k)\le d_k-d_{k-1},
\qquad
\sum_{j=1}^kR_j=(G_T)^{[d_k]},
$$

where $G_T=X^T(X^T)^\top$ and $(G_T)^{[d_k]}$ is its truncated spectral approximation.

Unlike a mini-batch eigendecomposition, this target is defined by corpus statistics and is not rank-limited by the optimization batch size.

## 3. Residual distance shells

Directly optimizing an $N\times N$ Gram matrix is unnecessary. For two corpus examples $x_n,x_m$, define the teacher contribution of spectral shell $k$ to their squared distance:

$$
r_k(n,m)
=
\left\|V_k^\top\left(q^T_n-q^T_m\right)\right\|_2^2
=
\left(q^T_n-q^T_m\right)^\top
P_k
\left(q^T_n-q^T_m\right).
$$

For a student view $q^S_n=q_\theta(a(x_n))$, define the contribution of coordinate band $I_k$:

$$
s_k(n,m)
=
\left\|
q^S_n[d_{k-1}+1:d_k]
-q^S_m[d_{k-1}+1:d_k]
\right\|_2^2.
$$

Both decompositions are exactly additive:

$$
\sum_{j=1}^ks_j(n,m)
=
\left\|q^S_n[1:d_k]-q^S_m[1:d_k]\right\|_2^2,
$$

$$
\sum_{j=1}^kr_j(n,m)
=
\left\|V_{1:d_k}^\top(q^T_n-q^T_m)\right\|_2^2.
$$

The GSR loss matches the increments rather than repeatedly matching whole prefixes:

$$
\boxed{
\mathcal L_{\mathrm{GSR}}
=
\frac1K\sum_{k=1}^K
\frac{
\mathbb E_{n\ne m}
\left[(s_k(n,m)-r_k(n,m))^2\right]
}{c_T+\varepsilon}
}
$$

with the fixed teacher scale

$$
c_T
=
\mathbb E_{n\ne m}
\left[
\|q^T_n-q^T_m\|_2^4
\right].
$$

A single common denominator preserves the relative energy of the spectral shells. It avoids both the scale ambiguity of CKA and the noise amplification caused by independently normalizing weak tail shells.

## 4. Unbiased mini-batch optimization

Let $\mathcal B=\{x_{b_1},\ldots,x_{b_B}\}$ be sampled uniformly without replacement. We estimate each shell loss using all unordered pairs:

$$
\widehat{\ell}_k(\mathcal B)
=
\frac{2}{B(B-1)}
\sum_{1\le a<b\le B}
\frac{
\left(s_k(b_a,b_b)-r_k(b_a,b_b)\right)^2
}{c_T+\varepsilon}.
$$

Then

$$
\mathbb E_{\mathcal B}
\left[\widehat{\ell}_k(\mathcal B)\right]
=
\frac{
\mathbb E_{n\ne m}
\left[(s_k(n,m)-r_k(n,m))^2\right]
}{c_T+\varepsilon}.
$$

Therefore

$$
\widehat{\mathcal L}_{\mathrm{GSR}}
=\frac1K\sum_k\widehat{\ell}_k
$$

is an unbiased degree-two U-statistic for the corpus pair objective. The batch size controls estimator variance, not the number or rank of meaningful shells. In particular, a batch of 16 can train targets at dimensions 16, 32, 64, and beyond without forcing later shells to zero.

This pair formulation also eliminates two ambiguities of mini-batch Gram matching: per-batch centering and the unequal sampling probabilities of Gram diagonal and off-diagonal entries.

## 5. Task objective and training protocol

For two student augmentations $a_1(x),a_2(x)$, retain the ordinary Matryoshka task objective:

$$
\mathcal L_{\mathrm{MRL}}
=
\sum_{k=1}^K\omega_k
\mathcal L_{\mathrm{task}}
\left(
\operatorname{norm}(q^{S,1}_{1:d_k}),
\operatorname{norm}(q^{S,2}_{1:d_k})
\right).
$$

The complete objective is

$$
\boxed{
\mathcal L
=
\mathcal L_{\mathrm{MRL}}
+\lambda\,
\frac12
\left[
\mathcal L_{\mathrm{GSR}}(q^{S,1},q^T)
+
\mathcal L_{\mathrm{GSR}}(q^{S,2},q^T)
\right].
}
$$

The teacher is deterministic; only the student receives stochastic augmentation. Gradients never pass through $q^T$, $V$, or $c_T$.

### Alternating teacher refresh

1. Warm up the student with $\mathcal L_{\mathrm{MRL}}$.
2. Snapshot $\bar\theta\leftarrow\operatorname{sg}(\theta)$.
3. Encode the corpus with $\bar\theta$; compute $V$, the cached teacher spectral coordinates $(Q^T-\mu_T)V$, and $c_T$.
4. Freeze all teacher quantities and optimize the student for one outer block or epoch.
5. Refresh the teacher and repeat.

Freezing the teacher within each outer block gives a well-defined optimization objective and avoids step-wise eigenvector chasing. The method needs no per-step eigendecomposition, projector, token selection, or inference-time component.

## 6. Mathematical properties

### Proposition 1: rank-matched global refinement

For each $k$, the cumulative teacher target is the best rank-$d_k$ approximation of the centered teacher Gram matrix in Frobenius norm:

$$
\sum_{j=1}^kR_j
=(G_T)^{[d_k]}
\in
\arg\min_{\operatorname{rank}(H)\le d_k}
\|G_T-H\|_F.
$$

Each increment $R_k$ has rank at most the width of its assigned student band.

**Reason.** This is the truncated eigendecomposition of $G_T$, grouped according to the Matryoshka boundaries.

### Proposition 2: local shell error controls every prefix

Let

$$
e_k(n,m)=s_k(n,m)-r_k(n,m).
$$

The squared-distance error at prefix $d_k$ satisfies

$$
\begin{aligned}
&\mathbb E_{n\ne m}
\left[
\left(
\|q^S_n[1:d_k]-q^S_m[1:d_k]\|_2^2
-
\|V_{1:d_k}^\top(q^T_n-q^T_m)\|_2^2
\right)^2
\right]\\
&\qquad=
\mathbb E_{n\ne m}
\left[\left(\sum_{j=1}^ke_j(n,m)\right)^2\right]\\
&\qquad\le
k\sum_{j=1}^k
\mathbb E_{n\ne m}[e_j(n,m)^2].
\end{aligned}
$$

Thus minimizing local shell errors simultaneously bounds the geometry distortion of every supported prefix. Opposing shell errors may cancel in a cumulative prefix error, but GSR does not reward that cancellation because it penalizes every increment separately.

### Proposition 3: residual supervision diagonalizes scale coupling

For one pair, collect the shell errors into

$$
e=[e_1,\ldots,e_K]^\top
$$

and let $L\in\mathbb R^{K\times K}$ be the cumulative-sum matrix, $L_{kj}=1$ if $j\le k$ and zero otherwise. A scale-sensitive loss that independently matches every rank-matched prefix has error geometry

$$
\ell_{\mathrm{prefix}}
=
\sum_{k=1}^K\left(\sum_{j=1}^ke_j\right)^2
=e^\top L^\top Le,
$$

whereas GSR has

$$
\ell_{\mathrm{shell}}=e^\top e.
$$

The two losses have the same exact zero set, but not the same optimization geometry. The dense matrix $L^\top L$ couples every early-band error to all later prefixes and has condition number $\Theta(K^2)$; shell supervision yields an identity metric in residual-error space. GSR therefore provides direct credit assignment to the responsible band instead of optimizing it through cumulative cross-scale interference.

### Proposition 4: block-subspace identifiability

Assume exact shell matching for every corpus pair and sufficient rank in shell $k$. After centering, there exists an orthogonal matrix $O_k\in\mathbb R^{\Delta d_k\times\Delta d_k}$ such that

$$
X^S[:,I_k]=X^TV_kO_k.
$$

Therefore GSR identifies the teacher principal subspace assigned to each coordinate band, while intentionally remaining invariant to rotations within that band.

This is a **block-privileged basis** result. It guarantees the requested endpoints $d_1,\ldots,d_K$, not arbitrary truncations inside a band. Coordinate-wise identifiability would require rank-one shells or additional within-band prefix supervision.

### Proposition 5: relationship to post-hoc PCA

For a frozen teacher, the construction

$$
X^S=[X^TV_1O_1\mid\cdots\mid X^TV_KO_K]
$$

achieves zero global GSR loss. Hence frozen-teacher GSR contains post-hoc PCA rotation as an exact solution.

This fact bounds the claim of the method: GSR is not a new PCA algorithm. Its learning hypothesis is that **joint MRL and residual-shell supervision co-adapt the representation geometry**, producing more task-useful spectral prefixes than applying PCA after conventional training. Post-hoc PCA and a frozen-encoder learned rotation are therefore mandatory baselines, not optional ablations.

## 7. Spectral non-uniqueness

The cumulative projector at boundary $d_k$ is unique only when

$$
\lambda_{d_k}>\lambda_{d_k+1}.
$$

GSR is invariant to signs and rotations inside a shell, but an eigenvalue tie crossing a Matryoshka boundary makes the split itself non-identifiable. We use the following rule:

- retain every task prefix in $\mathcal L_{\mathrm{MRL}}$;
- apply separate geometry shells only at boundaries with a resolved eigengap;
- merge adjacent geometry shells when a numerical tie crosses their boundary.

The guarantee in Proposition 2 then holds at the retained geometry boundaries. Near-ties should be reported through the relative boundary gap

$$
\gamma_k
=
\frac{\lambda_{d_k}-\lambda_{d_k+1}}
{\max(\lambda_{d_k},\varepsilon)}.
$$

This makes the method honest about what the teacher spectrum can identify rather than imposing an unstable arbitrary basis.

## 8. Complexity

Teacher refresh requires one corpus encoding pass and a covariance eigendecomposition. For large corpora, only the required leading eigenspaces need to be computed with randomized or incremental PCA.

Teacher spectral coordinates can be cached once per outer block. Per training batch, GSR computes band-wise pair distances with cost

$$
O(B^2D)
$$

and $O(B^2)$ working memory. There is no per-step SVD and no requirement that $B>d_k$. Inference is identical to ordinary MRL: truncate the embedding and normalize the selected prefix.

## 9. Defensible novelty boundary

The contribution is not “using PCA for Matryoshka embeddings.” PCA-guided compression and learned prefix transforms already exist.

The proposed contribution is the combination of:

1. **Residual assignment:** a one-to-one map from each nested coordinate band to a disjoint spectral increment of one global teacher geometry.
2. **Exact successive refinement:** student coordinate distances and teacher spectral distances admit parallel additive decompositions, so local band supervision controls every cumulative prefix.
3. **Decoupled credit assignment:** residual supervision diagonalizes the cumulative scale coupling induced by whole-prefix matching.
4. **Batch-rank-free stochastic training:** a pairwise U-statistic optimizes corpus geometry without constructing a corpus Gram matrix or requiring the batch size to exceed the prefix dimension.
5. **Joint co-adaptation:** standard task supervision remains active at every prefix while the geometry objective determines what each additional band contributes.

The nearest distinctions are:

- **MATE:** aligns every whole prefix to a PCA-compressed vector target using coordinate-level MSE/KL; it does not assign residual geometry to newly added bands or derive an unbiased pairwise geometry objective.
- **MIPIC:** aligns selected token-level full and truncated representations using scale-invariant CKA and attention-based selection; it does not decompose sentence-level geometry into rank-matched increments.
- **MIC:** discourages prefix/residual redundancy and spectral collapse; it provides a negative decorrelation constraint rather than a positive residual target.
- **SUGAR:** matches spectral capacity, subspaces, and band novelty in teacher-student distillation; GSR instead factorizes a single corpus geometry into exact Matryoshka rank increments and supervises their pairwise realization.
- **Full-prefix MRL theory:** explains how nested objectives induce a privileged basis; GSR explicitly supplies a global geometric target for each block and provides a stochastic shell-to-prefix guarantee.

## 10. Claims the paper should and should not make

### Supported by the formulation

- GSR removes the mini-batch rank ceiling of batch-Gram spectral supervision.
- Every band receives a capacity-matched residual geometry target.
- The proposed mini-batch estimator is unbiased for the corpus pair objective.
- Small shell errors upper-bound the distance distortion of every supported prefix.
- Exact matching identifies spectral blocks up to within-band orthogonal transforms.
- The method adds no inference overhead.

### Require empirical evidence

- Joint GSR produces better task prefixes than post-hoc PCA.
- Spectral residual assignment improves downstream accuracy or robustness.
- A particular refresh period, loss weight, or boundary schedule is optimal.
- Gains correlate with spectral tail energy or baseline shell leakage.

### Claims to avoid

- “The first PCA-guided Matryoshka method.”
- “The optimal low-dimensional representation” without naming the optimized geometry and norm.
- “Every coordinate is ordered” when only block endpoints are supervised.
- “Canonical spectral target” without an eigengap qualification.
- “Distillation” for a same-step target derived from the student itself.

## 11. Minimal falsifiable prediction

For a baseline MRL model, define shell leakage

$$
\operatorname{Leak}
=
1-
\frac1K\sum_{k=1}^K
\frac{
\mathbb E_{n\ne m}[s_k(n,m)r_k(n,m)]
}{
\sqrt{\mathbb E[s_k(n,m)^2]}
\sqrt{\mathbb E[r_k(n,m)^2]}
+\varepsilon
}.
$$

GSR predicts that reducing this off-assignment improves low-dimensional task quality when the teacher boundary eigengaps are stable. The method is falsified as a distinct training contribution if its gains are matched by post-hoc PCA or if lower shell error improves only the auxiliary geometry metric while leaving task prefixes unchanged.

## 12. Core paper statement

> We formulate Matryoshka representation learning as geometric successive refinement. A corpus-level teacher embedding induces a nested sequence of spectral subspaces, while Matryoshka coordinate bands induce an exactly additive sequence of pairwise distance increments. GSR assigns each coordinate band the residual spectral geometry unlocked by its additional width. A pairwise U-statistic provides unbiased mini-batch optimization without the rank ceiling of batch Gram matrices, and local shell errors provably control the geometry distortion of every supported prefix. Unlike whole-prefix PCA alignment or redundancy regularization, GSR supervises the missing geometry contributed by each newly added band while preserving task learning at every prefix.

## References for positioning

- Kusupati et al., *Matryoshka Representation Learning*, NeurIPS 2022. <https://arxiv.org/abs/2205.13147>
- Jung et al., *MATE: Matryoshka Audio-Text Embeddings for Open-Vocabulary Keyword Spotting*, 2026. <https://arxiv.org/abs/2601.14012>
- Phung et al., *MIPIC: Matryoshka Representation Learning via Self-Distilled Intra-Relational and Progressive Information Chaining*, 2026. <https://arxiv.org/abs/2604.24374>
- Talukder et al., *Objective-Specific Privileged Bases via Full-Prefix Matryoshka Learning*, 2026. <https://arxiv.org/abs/2605.09160>
- *MIC: Maximizing Informational Capacity in Adaptive Representations via Isotropic Subspace Alignment*, 2026. <https://arxiv.org/abs/2605.29987>
- *GraSP-VL: Length as a Semantic Granularity Interface for Vision-Language Representations*, 2026. <https://arxiv.org/abs/2605.17727>
