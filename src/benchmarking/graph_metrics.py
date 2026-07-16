"""APLS — Average Path Length Similarity (Van Etten et al. 2019, SpaceNet).

Compares the road *graphs* of a stitched binary prediction and its ground
truth, so it scores network connectivity rather than pixel overlap: a small
gap that pixel F1 barely notices severs every shortest path crossing it.

Pipeline (all in the tile's ground units via the GT-resolution transform):

    mask ──skeletonize──▶ 1-px skeleton ──chain-trace──▶ nx.Graph
                                                    (edges carry the pixel
                                                     polyline + length in m)

    APLS  = hmean(C_gt→prop, C_prop→gt)                    (CosmiQ convention)
    C_s→t = 1 − mean over control-node pairs (a,b) of
              min(1, |L_s(a,b) − L_t(a′,b′)| / L_s(a,b))
    where a′/b′ are a/b snapped onto the target graph (≤ SNAP_DIST_M, else the
    pair scores the full penalty 1), and control nodes are the source graph's
    junctions/endpoints plus points injected every CONTROL_DELTA_M along edges
    (so long uninterrupted roads still contribute path samples).

Design notes:
  * Pure numpy/scipy/networkx/skimage — deliberately NO sknw dependency
    (it drags in numba, which the bench venv doesn't carry). The chain tracer
    below produces the same junction/endpoint graph.
  * Skeletonization noise (spurs, jitter) is systematic: it hits every model's
    prediction and the GT identically, so *between-model* comparisons stay
    fair even though absolute values carry skeleton artefacts.
  * Determinism: node ordering is sorted, subsampling is seeded — the same
    (pred, gt) pair always yields the same score.
  * Empty handling mirrors ``pixel_metrics_from_counts``: GT and prediction
    both road-free → NaN (undefined, dropped by the paired stats); exactly one
    empty → 0.0 (total connectivity failure / pure hallucination).

GSD-aware defaults (metres, converted to pixels via the transform):
  SNAP_DIST_M      30 — 3 px @ 10 m (masks share a grid; skeletons land within
                        a pixel or two of each other when the roads match)
  CONTROL_DELTA_M 500 — 50 px @ 10 m control-point spacing along edges
  MIN_SPUR_M       30 — prune dead-end skeleton spurs shorter than this
                        (3 px @ 10 m, matching the legacy graph extractor)
  MAX_NODES       200 — per-direction control-node cap (seeded subsample)
"""
from __future__ import annotations

import numpy as np

SNAP_DIST_M = 30.0
CONTROL_DELTA_M = 500.0
MIN_SPUR_M = 30.0
MAX_NODES = 200
_EPS_M = 1e-9

_OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


# --------------------------------------------------------------------------- #
# mask -> graph (skeleton chain tracing)
# --------------------------------------------------------------------------- #
def _polyline_len_m(pts: np.ndarray, px_m: float) -> float:
    """Length of an (N,2) pixel polyline in metres."""
    if len(pts) < 2:
        return 0.0
    d = np.diff(np.asarray(pts, dtype=float), axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum()) * px_m


def _add_chain(G, chain: list[tuple[int, int]], px_m: float) -> None:
    """Add a traced pixel chain as an edge u—v (node ids ARE (row, col) tuples).

    ``pts[0]``/``pts[-1]`` always equal the endpoint node ids — the splitters
    below rely on that invariant. Parallel chains between the same node pair
    keep the SHORTER one (exactly what any shortest path would use).
    """
    u, v = chain[0], chain[-1]
    pts = np.asarray(chain, dtype=float)
    length = max(_polyline_len_m(pts, px_m), _EPS_M)
    if G.has_edge(u, v) and u != v and G[u][v]["length"] <= length:
        return
    G.add_node(u, o=np.asarray(u, dtype=float))
    G.add_node(v, o=np.asarray(v, dtype=float))
    G.add_edge(u, v, pts=pts, length=length)


def mask_to_graph(mask: np.ndarray, px_m: float, min_spur_m: float = MIN_SPUR_M):
    """(H, W) binary mask -> undirected road graph.

    Skeletonizes the mask (Zhang-Suen, 1-px centreline), then traces
    8-connected pixel chains between the skeleton's junction/endpoint pixels
    (degree != 2). Node ids are (row, col) pixel tuples with attr ``o`` (float
    coords); edges carry ``pts`` (the (N,2) pixel polyline INCLUDING both
    endpoint pixels) and ``length`` (metres). Pure cycles with no junction
    (isolated rings) become a self-loop anchored at their smallest pixel.
    Dead-end spurs shorter than ``min_spur_m`` are pruned once (skeleton
    artefacts of jagged mask edges), then isolated nodes are dropped.
    """
    import networkx as nx
    from skimage.morphology import skeletonize

    G = nx.Graph()
    G.graph["px_m"] = px_m
    mask = np.asarray(mask) > 0
    if not mask.any():
        return G
    skel = skeletonize(mask)
    coords = {(int(r), int(c)) for r, c in zip(*np.nonzero(skel))}
    if not coords:
        return G

    def nbrs(p):
        r, c = p
        return [(r + dr, c + dc) for dr, dc in _OFFSETS if (r + dr, c + dc) in coords]

    neighbours = {p: nbrs(p) for p in coords}
    node_px = {p for p, ns in neighbours.items() if len(ns) != 2}

    stepped: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    chained: set[tuple[int, int]] = set()

    # chains between junction/endpoint pixels
    for p in sorted(node_px):
        for q in sorted(neighbours[p]):
            if (p, q) in stepped:
                continue
            chain = [p]
            prev, cur = p, q
            while True:
                stepped.add((prev, cur))
                stepped.add((cur, prev))
                chain.append(cur)
                chained.add(cur)
                if cur in node_px:
                    break
                nxt = [t for t in neighbours[cur] if t != prev]
                if not nxt:  # safety: degree-2 bookkeeping can't fail, but be robust
                    break
                prev, cur = cur, nxt[0]
            if len(chain) >= 2:
                _add_chain(G, chain, px_m)

    # pure cycles (rings with every pixel at degree 2 -> never reached above)
    remaining = sorted(p for p in coords if len(neighbours[p]) == 2 and p not in chained
                       and p not in node_px)
    seen = set(chained) | node_px
    for p0 in remaining:
        if p0 in seen:
            continue
        chain = [p0]
        seen.add(p0)
        prev, cur = p0, sorted(neighbours[p0])[0]
        while cur != p0:
            chain.append(cur)
            seen.add(cur)
            nxt = [t for t in neighbours[cur] if t != prev]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
        chain.append(p0)  # close the loop
        if len(chain) >= 3:
            _add_chain(G, chain, px_m)

    # prune short dead-end spurs, then drop what they leave isolated
    spurs = [n for n in G.nodes
             if G.degree(n) == 1
             and next(iter(G.edges(n, data=True)))[2]["length"] < min_spur_m]
    G.remove_nodes_from(spurs)
    G.remove_nodes_from(list(nx.isolates(G)))
    return G


# --------------------------------------------------------------------------- #
# edge splitting (shared by control injection and snap insertion)
# --------------------------------------------------------------------------- #
def _oriented_pts(G, u, v) -> np.ndarray:
    """Edge polyline oriented so pts[0] is u's pixel (node ids are px tuples)."""
    pts = G[u][v]["pts"]
    if tuple(int(x) for x in pts[0]) == tuple(u):
        return pts
    return pts[::-1]


def _split_edge_at_indices(G, u, v, indices) -> dict[int, tuple[int, int]]:
    """Split edge u—v at the given polyline indices (u-oriented), in place.

    Returns {polyline_index: node_id} for every requested index (indices that
    fall on the endpoints map to u/v themselves). New node ids are the pixel
    tuples of the split vertices — chain interiors are unique to one edge, so
    ids never collide. Sub-edges keep exact sub-polylines and lengths, so
    shortest paths through the split points are exact.
    """
    px_m = G.graph["px_m"]
    pts = _oriented_pts(G, u, v)
    last = len(pts) - 1
    mapping: dict[int, tuple[int, int]] = {}
    interior = sorted({int(i) for i in indices if 0 < int(i) < last})
    for i in indices:
        i = int(i)
        if i <= 0:
            mapping[i] = u
        elif i >= last:
            mapping[i] = v
    if not interior:
        return mapping

    bounds = [0, *interior, last]
    node_ids = [u] + [tuple(int(x) for x in pts[i]) for i in interior] + [v]
    for i, nid in zip(interior, node_ids[1:-1]):
        mapping[i] = nid

    G.remove_edge(u, v)
    for (i0, i1), (a, b) in zip(zip(bounds[:-1], bounds[1:]),
                                zip(node_ids[:-1], node_ids[1:])):
        sub = pts[i0:i1 + 1]
        length = max(_polyline_len_m(sub, px_m), _EPS_M)
        if G.has_edge(a, b) and a != b and G[a][b]["length"] <= length:
            continue
        G.add_node(a, o=np.asarray(sub[0], dtype=float))
        G.add_node(b, o=np.asarray(sub[-1], dtype=float))
        G.add_edge(a, b, pts=np.asarray(sub), length=length)
    return mapping


def _inject_controls(G, delta_m: float):
    """Copy of G with nodes injected every ~delta_m of arc length along each
    edge (Van Etten's control points): path endpoints then sample edge
    interiors, not just junctions. Self-loops are cut at >= 2 points so an
    isolated ring still yields node pairs."""
    G2 = G.copy()
    px_m = G.graph["px_m"]
    for u, v, data in list(G.edges(data=True)):
        pts = data["pts"]
        n_seg = int(np.ceil(data["length"] / delta_m))
        if u == v:
            n_seg = max(n_seg, 3)
        if n_seg < 2:
            continue
        cum = np.concatenate([[0.0], np.cumsum(
            np.hypot(*np.diff(np.asarray(pts, dtype=float), axis=0).T))]) * px_m
        cuts = [int(np.searchsorted(cum, data["length"] * k / n_seg))
                for k in range(1, n_seg)]
        _split_edge_at_indices(G2, u, v, cuts)
    return G2


# --------------------------------------------------------------------------- #
# directional APLS
# --------------------------------------------------------------------------- #
def _directional_apls(G_src, G_tgt, snap_dist_m: float, control_delta_m: float,
                      max_nodes: int, rng) -> float | None:
    """C_src→tgt = 1 − mean pair penalty. None = undefined (no source pairs)."""
    import networkx as nx
    from scipy.spatial import cKDTree

    if G_src.number_of_edges() == 0:
        return None
    if G_tgt.number_of_edges() == 0:
        return 0.0
    px_m = G_src.graph["px_m"]

    Gs = _inject_controls(G_src, control_delta_m)
    controls = sorted(Gs.nodes)
    if len(controls) > max_nodes:
        keep = rng.choice(len(controls), size=max_nodes, replace=False)
        controls = [controls[i] for i in sorted(keep)]

    # snap every control point onto the target graph's polylines
    flat_pts, refs = [], []
    for u, v, data in G_tgt.edges(data=True):
        pts = _oriented_pts(G_tgt, u, v)
        for i, p in enumerate(pts):
            flat_pts.append(p)
            refs.append((u, v, i))
    tree = cKDTree(np.asarray(flat_pts, dtype=float))
    dists, hits = tree.query(np.asarray([Gs.nodes[c]["o"] for c in controls]))

    Gt = G_tgt.copy()
    cuts_per_edge: dict[tuple, list[int]] = {}
    pending: dict = {}
    for c, d_px, hit in zip(controls, dists, hits):
        if d_px * px_m > snap_dist_m:
            pending[c] = None
            continue
        u, v, i = refs[hit]
        pending[c] = (u, v, i)
        cuts_per_edge.setdefault((u, v), []).append(i)

    snapped: dict = {}
    for (u, v), idxs in cuts_per_edge.items():
        mapping = _split_edge_at_indices(Gt, u, v, idxs)
        for c, ref in pending.items():
            if ref is not None and ref[0] == u and ref[1] == v:
                snapped[c] = mapping[int(ref[2])]
    for c, ref in pending.items():
        if ref is None:
            snapped[c] = None

    # pairwise path-length comparison
    penalty_sum, n_pairs = 0.0, 0
    for ai, a in enumerate(controls[:-1]):
        len_src = nx.single_source_dijkstra_path_length(Gs, a, weight="length")
        ta = snapped[a]
        len_tgt = (nx.single_source_dijkstra_path_length(Gt, ta, weight="length")
                   if ta is not None else {})
        for b in controls[ai + 1:]:
            L_s = len_src.get(b)
            if L_s is None or L_s <= 0:
                continue  # APLS counts only pairs with a path in the source
            n_pairs += 1
            tb = snapped[b]
            if ta is None or tb is None:
                penalty_sum += 1.0
                continue
            L_t = len_tgt.get(tb)
            if L_t is None:
                penalty_sum += 1.0  # both snapped but disconnected in target
                continue
            penalty_sum += min(1.0, abs(L_s - L_t) / L_s)
    if n_pairs == 0:
        return None
    return float(np.clip(1.0 - penalty_sum / n_pairs, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #
def apls_tile(pred_bin: np.ndarray, gt_mask: np.ndarray, *, transform=None,
              snap_dist_m: float = SNAP_DIST_M,
              control_delta_m: float = CONTROL_DELTA_M,
              min_spur_m: float = MIN_SPUR_M,
              max_nodes: int = MAX_NODES,
              seed: int = 0) -> dict[str, float]:
    """APLS between one tile's stitched prediction and ground truth.

    Returns tile-level columns: ``apls`` (harmonic mean of the two directions,
    the headline number), the directional scores, and graph sizes for
    diagnosing skeleton quality. NaN where undefined (both masks road-free);
    0.0 when exactly one side has roads.
    """
    px_m = abs(transform.a) if transform is not None else 1.0
    if not np.isfinite(px_m) or px_m <= 0:
        px_m = 1.0

    G_gt = mask_to_graph(gt_mask, px_m, min_spur_m)
    G_pr = mask_to_graph(pred_bin, px_m, min_spur_m)
    rng = np.random.default_rng(seed)

    c_gp = _directional_apls(G_gt, G_pr, snap_dist_m, control_delta_m, max_nodes, rng)
    c_pg = _directional_apls(G_pr, G_gt, snap_dist_m, control_delta_m, max_nodes, rng)

    if c_gp is None and c_pg is None:
        apls = float("nan")
    elif c_gp is None:
        apls = float(c_pg)
    elif c_pg is None:
        apls = float(c_gp)
    elif c_gp <= 0.0 or c_pg <= 0.0:
        apls = 0.0
    else:
        apls = 2.0 * c_gp * c_pg / (c_gp + c_pg)  # harmonic mean (CosmiQ apls)

    nan = float("nan")
    return {
        "apls": apls,
        "apls_gt_to_prop": nan if c_gp is None else float(c_gp),
        "apls_prop_to_gt": nan if c_pg is None else float(c_pg),
        "gt_graph_nodes": float(G_gt.number_of_nodes()),
        "gt_graph_edges": float(G_gt.number_of_edges()),
        "prop_graph_nodes": float(G_pr.number_of_nodes()),
        "prop_graph_edges": float(G_pr.number_of_edges()),
    }
