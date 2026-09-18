# Probe constructions — full textual and mathematical definitions

Companion to `docs/lda_cka_band_probes_plan.md`. Everything an implementation
or a thesis appendix needs to reproduce the three instruments exactly.

## 0. Notation and shared objects

* Chips: the fixture set $C$, $|C| = 400$ test chips. A chip $c$ is one
  128 px LR window at 10 m; a model's forward maps it to a 512 px output at
  2.5 m.
* For arm $a$, seed $s$: the model $M_{a,s}$. Its SR front-end output on
  chip $c$ is
  $$y_{a,s}(c) \in \mathbb{R}^{4 \times 512 \times 512},$$
  the **pre-adapter** tensor in reflectance units (each checkpoint is fed
  its own raw units and read back through its own `reflectance_scale`; this
  is the exact tensor its U-Net consumes, before the z-score). Bands are
  indexed $j \in \{B, G, R, NIR\}$.
* The z-score adapter of $M_{a,s}$ has frozen (post-recalibration) buffers
  $\mu_{a,s}, \sigma_{a,s} \in \mathbb{R}^4$; the U-Net input is
  $x_{seg} = (y - \mu_{a,s}) / \sigma_{a,s}$.
* Pixel sample: a fixed index set $P = P_{road} \cup P_{bg}$ of
  (chip, row, col) triples at the 2.5 m grid, $|P_{road}| = 10{,}000$,
  $|P_{bg}| = 40{,}000$, drawn pooled-proportionally once and reused for
  every arm — so pixel $p$ names the *same location* in every arm's output.
  Class labels come from the 2.5 m rasterised mask.
* For a pixel $p$ and model $(a,s)$, the observation is the 4-vector of
  band values $x_p^{(a,s)} = y_{a,s}(c_p)[:, r_p, k_p] \in \mathbb{R}^4$.

r0's "SR output" is the bicubic upsample — parameter-free, so
$x_p^{(r0,s)}$ is identical for every seed; r0 statistics are computed once.

---

## 1. LDA in a shared frame

### 1.1 The fit (on r0 only)

Compute class means and the pooled within-class scatter on r0's pixels:
$$\mu_k = \frac{1}{|P_k|} \sum_{p \in P_k} x_p, \qquad k \in \{road, bg\}$$
$$S_W = \sum_{k} \sum_{p \in P_k} (x_p - \mu_k)(x_p - \mu_k)^\top
\in \mathbb{R}^{4\times4}.$$

The two-class Fisher discriminant direction is
$$w_0 = S_W^{-1}(\mu_{road} - \mu_{bg}), \qquad \hat w_0 = w_0 / \lVert w_0 \rVert.$$

This single unit vector is **the shared frame**. LD1 coordinate of any
pixel from any arm: $z_p = \hat w_0^\top x_p$.

Second plot axis: project the data onto the orthogonal complement of
$\hat w_0$, i.e. $r_p = x_p - (\hat w_0^\top x_p)\,\hat w_0$, pool both
classes, and take the top principal component $v$ of
$\mathrm{Cov}(\{r_p\})$. By construction $v \perp \hat w_0$, so
$(\hat w_0^\top x_p,\; v^\top x_p)$ is an orthonormal 2-D projection.

Loadings for display: report $\hat w_{0,j}\,\sigma_j^{(r0)}$ per band
(contribution per standard deviation of that band in r0), not the raw
components — raw reflectance scales differ across bands and would make the
raw components unreadable.

### 1.2 The scores

For any arm-seed, compute its own class means $\mu_k^{(a,s)}$ and pooled
within-class covariance $\Sigma_W^{(a,s)}$ from its pixel values. The
Fisher ratio along a direction $u$ is
$$J_u(a,s) = \frac{\left(u^\top(\mu_{road}^{(a,s)} - \mu_{bg}^{(a,s)})\right)^2}
{u^\top \Sigma_W^{(a,s)} u}.$$

Two numbers per arm-seed:

* **r0-frame Fisher:** $J_{\hat w_0}(a,s)$ — separability along the
  direction that already separated the un-enhanced data.
* **Own-frame Fisher:** $\max_u J_u(a,s) = (\mu_{road} - \mu_{bg})^\top \Sigma_W^{-1} (\mu_{road} -\mu_{bg})$ ,
  the squared Mahalanobis distance between the class means (means and
  $\Sigma_W$ from that arm-seed). This is the supremum of $J$ over all
  directions, attained at $u = \Sigma_W^{-1}(\mu_{road} - \mu_{bg})$.

Two properties the reading depends on:

1. $J_{own} \ge J_{\hat w_0}$ always. The **gap** measures how far the
   arm's optimal discriminant has rotated away from r0's — "created a new
   direction" vs "sharpened the existing one".
2. $J_{own}$ is invariant under any invertible linear map of the bands
   (Mahalanobis affine-invariance), so it measures *intrinsic* linear
   separability and is blind to per-band affine drift — the same blindness
   the 1×1 probe exploits. $J_{\hat w_0}$ is deliberately **not**
   affine-invariant: the frame is pinned, so radiometric walks move it.
   The pair (invariant, pinned) is what separates cosmetic from structural
   change.

### 1.3 The figure inputs

Ridgeline: per arm, the histograms of $\{z_p : p \in P_{road}\}$ and
$\{z_p : p \in P_{bg}\}$ on a common LD1 axis. Slopegraph: the pair
$(J_{\hat w_0}, J_{own})$ per arm-seed.

---

## 2. Stage-wise linear CKA

### 2.1 Activations

Stage set $L$: the post-activation output of `conv1` and encoder stages
1–4 of the ResNet-34 (the feature pyramid the decoder consumes), plus each
decoder block. At stage $\ell$ the activation on chip $c$ is a tensor
$A_\ell(c) \in \mathbb{R}^{C_\ell \times H_\ell \times W_\ell}$
(resolutions halve per encoder stage: 256², 128², 64², 32², 16² for a
512 px input).

Examples for CKA are **(chip, spatial position)** pairs: at each stage,
fix a position subsample $G_\ell$ of up to $m$ = 2000 positions per chip
(all positions when $H_\ell W_\ell \le m$), the same normalized positions
for every model, sampled once. Stack rows over chips and positions to get
the data matrix
$$X_\ell \in \mathbb{R}^{n_\ell \times C_\ell}, \qquad
n_\ell = |C| \cdot \min(m, H_\ell W_\ell),$$
rows paired across models (row $i$ = same chip, same position).

### 2.2 The statistic

Column-center both matrices ($\tilde X = X - \mathbf{1}\bar x^\top$).
Linear CKA between models $A$ and $B$ at stage $\ell$:
$$\mathrm{CKA}(X, Y) =
\frac{\lVert \tilde Y^\top \tilde X \rVert_F^2}
{\lVert \tilde X^\top \tilde X \rVert_F \; \lVert \tilde Y^\top \tilde Y \rVert_F}.$$

This is the feature-space form: it needs only $C_X \times C_Y$
cross-covariance matrices, cost $O(n C^2)$, no $n \times n$ Gram matrix —
exact, not an approximation, for the *linear* kernel. (The minibatch HSIC
estimator from the literature is only needed for RBF kernels or when even
$X$ doesn't fit in memory.)

Properties: invariant to orthogonal transforms and isotropic scaling of
either feature space (hence robust to channel permutation — why separately
trained nets are comparable), **not** invariant to general invertible
linear maps. Range (0, 1], higher = more similar.

### 2.3 The two stimulus conventions

Both models in a pair are always evaluated on the **same chips**; what
varies is which image each network sees:

1. **Own-pipeline:** network $N_{a,s}$ runs on its own pipeline's input
   $x_{seg}^{(a,s)}(c) = (y_{a,s}(c) - \mu_{a,s})/\sigma_{a,s}$. The
   comparison mixes input-induced and weight-induced differences — the
   deployed situation.
2. **Common-input:** every network runs on r0's bicubic reflectance
   $y_{r0}(c)$, passed through **that network's own** frozen z-score
   $(y_{r0}(c) - \mu_{a,s})/\sigma_{a,s}$. The adapter is part of the
   arm's front-end; freezing it out would feed the U-Net a distribution it
   never trained on. With the image pinned, remaining divergence is in the
   weights.

The per-stage difference between convention 1 and convention 2 CKA is the
input-inherited share of the divergence.

### 2.4 Pairs and the noise floor

* Between-arm, at matched stage: all seed pairs $(s, s')$ of $(a, a')$;
  report the mean, show the spread.
* Within-arm floor: all $\binom{k}{2}$ seed pairs of the same arm
  (currently only r2a has $k \ge 3$; its pooled per-stage floor is the
  grey band, labelled with its provenance).
* Decision rule: a between-arm CKA is only interpreted where it falls
  **below** the floor band at that stage.

---

## 3. Band occlusion

### 3.1 The intervention

For arm-seed $(a,s)$, chip $c$, and condition
$k \in \{R, G, B, NIR, RGB\}$ with band set $B_k$: build
$$\tilde y(c)[j] = \begin{cases}
\mu_{a,s}[j] \cdot \mathbf{1} & j \in B_k \\
y_{a,s}(c)[j] & j \notin B_k
\end{cases}$$
i.e. replace each occluded band by that model's own adapter mean,
**pre-adapter**. After the z-score the occluded band is exactly
$(\mu - \mu)/\sigma = 0$: the U-Net receives its trained notion of "this
band at its average, everywhere" — no information, no distribution shock
in that channel's first moment. Then run the U-Net forward unchanged.

Five conditions are complete and non-redundant for a 4-band model: the
four single bands, plus RGB as the complement of NIR.

### 3.2 The readouts

On the fixture chips (all pixels, natural distribution, empty chips
included), against the 2.5 m ground truth:

* $\mathrm{AP}$: average precision of the pixel scores
  $\sigma(\text{logits})$, micro-pooled over all pixels of all fixture
  chips. $\Delta \mathrm{AP}_k = \mathrm{AP}(\tilde y_k) - \mathrm{AP}(y)$.
  Primary, threshold-free.
* $\mathrm{IoU}@\theta^*$: binarise at the run's own $\theta^*$
  (re-argmaxed from its `sweep.json` under the micro-IoU criterion),
  micro-pooled road-class IoU; $\Delta \mathrm{IoU}_k$ analogous.
  Excluded for any run whose $\theta^*$ is a fallback.
* Relative drop $\Delta \mathrm{AP}_k / \mathrm{AP}$ annotated where
  clean baselines differ strongly between arms.

Both are **probe-local** (computed on 128 px windows) and must never be
compared against benchmark-store rows.

### 3.3 What it measures, and the check

Mean-substitution measures **reliance**, not information content: bands
are correlated, so a network may recover part of an occluded band from the
others, and a constant band is still mildly off-manifold. The pre-declared
robustness variant is **band permutation**: replace band $j$ of chip $c$
with band $j$ of a different chip (a random derangement over the fixture),
which preserves the band's marginal distribution while destroying its
spatial alignment. Per the plan's §2 pre-commitment, if this variant is
run *because* a mean-substitution result was surprising, it is labelled
exploratory.

---

## 4. What each instrument can and cannot claim

* **LDA** speaks about the SR *output* (U-Net input space). It is linear:
  a separability gain invisible to any linear projection (e.g. texture)
  will not register. That is what the 1×1-probe decodability and the
  U-Net itself are for.
* **CKA** speaks about *representations*, not performance: two networks
  can differ in CKA and segment identically, or vice versa. It localises
  where change happened; it does not grade the change.
* **Occlusion** speaks about the *trained network's use* of its input
  channels under intervention, on-distribution-except-one-band. It is the
  only one of the three that is causal with respect to the network's
  forward pass.
