"""
This is deprecated!

Skeleton-based road-graph extraction (mask -> SAM-Road graph dict).

A standalone converter: it derives a graph from a *raster mask* via
skeletonization (not from road vectors, so it is independent of the road-vector
pipeline). Operates on COG/GeoTIFF or PNG/JPG image+mask pairs. IO/stretch
helpers now live in :mod:`sentinel2data.generator.helper`
"""
import glob
import os
import pickle
from dataclasses import dataclass
import cv2
import numpy as np
import sknw
from skimage.morphology import skeletonize
from sentinel2data.generator.io import load_binary_mask, load_image_rgb

IMAGE_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


# --------------------------------------------------------------------------- #
# graph transforms (operate on numpy arrays / networkx graphs)
# --------------------------------------------------------------------------- #
def prune_graph_spurs(nx_graph, min_spur_length):
    """Remove short dead-end branches (spurs) that are skeletonization
    artifacts of jagged pixelated edges."""
    dead_ends = [n for n, d in nx_graph.degree() if d == 1]
    nodes_to_remove = []

    for n in dead_ends:
        neighbor = list(nx_graph.neighbors(n))[0]
        edge_data = nx_graph.get_edge_data(n, neighbor)

        if "weight" in edge_data:
            length = edge_data["weight"]
        else:
            length = len(edge_data.get("pts", []))

        if length < min_spur_length:
            nodes_to_remove.append(n)

    nx_graph.remove_nodes_from(nodes_to_remove)
    return nx_graph


def create_sam_road_graph(nx_graph, step_size):
    """Convert a networkx graph into the SAM-Road/CityScale dictionary format.

    Coordinates stay in the (upscaled) raster space, in (row, col) order.
    """
    sat2graph_dict = {}
    node_coords = {}

    for node_id, node_data in nx_graph.nodes(data=True):
        # sknw natively returns (row, col)
        r, c = int(node_data["o"][0]), int(node_data["o"][1])

        node_coords[node_id] = (r, c)
        if (r, c) not in sat2graph_dict:
            sat2graph_dict[(r, c)] = []

    for u, v, edge_data in nx_graph.edges(data=True):
        coord_u = node_coords[u]
        coord_v = node_coords[v]

        # The curve pixel coordinates (row, col)
        path = [(int(r), int(c)) for r, c in edge_data["pts"]]

        # Flow direction check
        if len(path) > 0:
            dist_u_start = (coord_u[0] - path[0][0]) ** 2 + (
                coord_u[1] - path[0][1]
            ) ** 2
            dist_u_end = (coord_u[0] - path[-1][0]) ** 2 + (
                coord_u[1] - path[-1][1]
            ) ** 2
            if dist_u_end < dist_u_start:
                path.reverse()

        # Walk the path and sample points
        sampled_nodes = [coord_u]
        for pt in path:
            last_pt = sampled_nodes[-1]
            dist = np.sqrt((pt[0] - last_pt[0]) ** 2 + (pt[1] - last_pt[1]) ** 2)
            if dist >= step_size:
                sampled_nodes.append(pt)

        # Ensure the final node connects perfectly to the intersection 'v'
        if sampled_nodes[-1] != coord_v:
            sampled_nodes.append(coord_v)

        # Add bidirectional connections in (row, col)
        for i in range(len(sampled_nodes) - 1):
            n1 = sampled_nodes[i]
            n2 = sampled_nodes[i + 1]

            if n1 not in sat2graph_dict:
                sat2graph_dict[n1] = []
            if n2 not in sat2graph_dict:
                sat2graph_dict[n2] = []

            if n2 not in sat2graph_dict[n1]:
                sat2graph_dict[n1].append(n2)
            if n1 not in sat2graph_dict[n2]:
                sat2graph_dict[n2].append(n1)

    return sat2graph_dict


def create_overlay(img, graph_dict):
    """Draw the graph dictionary onto a BGR image for visual verification."""
    overlay = img.copy()

    # Draw edges (white lines)
    for r1, c1 in graph_dict:
        for r2, c2 in graph_dict[(r1, c1)]:
            # cv2 requires (x, y) which is (col, row)
            cv2.line(overlay, (c1, r1), (c2, r2), (255, 255, 255), 2)

    # Draw nodes (green circles)
    for r, c in graph_dict:
        cv2.circle(overlay, (c, r), 4, (0, 255, 0), -1)

    return overlay


def create_keypoint_mask(nx_graph, shape, radius=3):
    """Render intersection/endpoint nodes (degree != 2) as filled dots."""
    keypoint_mask = np.zeros(shape, dtype=np.uint8)
    for node, degree in nx_graph.degree():
        if degree != 2:
            r, c = int(nx_graph.nodes[node]["o"][0]), int(nx_graph.nodes[node]["o"][1])
            cv2.circle(keypoint_mask, (c, r), radius, 255, -1)
    return keypoint_mask


# --------------------------------------------------------------------------- #
# Tile-level orchestration (arrays in, artifacts out -- no disk IO)
# --------------------------------------------------------------------------- #
@dataclass
class TileArtifacts:
    graph_dict: dict
    clean_mask: np.ndarray
    image: np.ndarray
    overlay: np.ndarray
    keypoint_mask: np.ndarray


def tile_to_graph(
    image_bgr,
    binary_mask,
    node_spacing=5,
    kernel_size=3,
    scale_factor=2.0,
    min_spur_length=3,
):
    """Convert one image/mask pair (already loaded as arrays) into graph
    artifacts. Pure: no file IO, so it is trivially testable and reusable
    across COG, GeoTIFF and PNG sources."""

    # 1. Upscale image + mask
    if scale_factor > 1.0:
        h, w = binary_mask.shape
        new_size = (int(w * scale_factor), int(h * scale_factor))
        # Masks use nearest neighbor to stay binary
        binary_mask = cv2.resize(binary_mask, new_size, interpolation=cv2.INTER_NEAREST)
        # RGB uses cubic for better visual upscaling
        image = cv2.resize(image_bgr, new_size, interpolation=cv2.INTER_CUBIC)
    else:
        image = image_bgr.copy()

    # 2. Morphological closing
    if kernel_size > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    # Output the "clean lines" mask from the morphologically closed mask
    clean_line_mask = binary_mask.copy()

    # 3. Skeletonize (required for sknw to extract the graph)
    skeleton = skeletonize(binary_mask > 0)

    # 4. Extract graph
    graph = sknw.build_sknw(skeleton.astype(np.uint16), multi=False)

    # 5. Prune noisy spurs (scaled to the new resolution)
    if min_spur_length > 0:
        graph = prune_graph_spurs(
            graph, min_spur_length=(min_spur_length * scale_factor)
        )

    # 6. Keypoint mask (intersections + endpoints)
    keypoint_mask = create_keypoint_mask(graph, binary_mask.shape)

    # 7. SAM-Road dictionary (node_spacing scaled to the new resolution)
    graph_dict = create_sam_road_graph(graph, step_size=(node_spacing * scale_factor))

    # 8. Visual overlay
    overlay_img = create_overlay(image, graph_dict)

    return TileArtifacts(
        graph_dict=graph_dict,
        clean_mask=clean_line_mask,
        image=image,
        overlay=overlay_img,
        keypoint_mask=keypoint_mask,
    )


# --------------------------------------------------------------------------- #
# Dataset-level driver
# --------------------------------------------------------------------------- #
def _find_matching_image(img_dir, base_name):
    """Locate the source image for a mask, trying known raster/png extensions."""
    for ext in IMAGE_EXTS:
        candidate = os.path.join(img_dir, base_name + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def convert_dataset_to_graphs(
    img_dir,
    mask_dir,
    output_dir,
    node_spacing=5,
    kernel_size=3,
    scale_factor=2.0,
    min_spur_length=3,
):
    """Process a directory of image/mask pairs (COG, GeoTIFF or PNG) into
    graph dicts plus debug rasters."""
    dirs = {
        "graphs": os.path.join(output_dir, "graphs_p"),
        "clean_masks": os.path.join(output_dir, "clean_masks"),
        "images": os.path.join(output_dir, "images"),
        "overlays": os.path.join(output_dir, "overlays"),
        "keypoint_masks": os.path.join(output_dir, "keypoint_masks"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    mask_paths = []
    for ext in IMAGE_EXTS:
        mask_paths.extend(glob.glob(os.path.join(mask_dir, "*" + ext)))
    print(f"Found {len(mask_paths)} masks to process in '{mask_dir}'.")

    for m_path in mask_paths:
        img_name = os.path.splitext(os.path.basename(m_path))[0]

        i_path = _find_matching_image(img_dir, img_name)
        if i_path is None:
            print(
                f"Warning: Could not find matching source image for {img_name}. Skipping."
            )
            continue

        try:
            image = load_image_rgb(i_path)
            mask = load_binary_mask(m_path)

            artifacts = tile_to_graph(
                image_bgr=image,
                binary_mask=mask,
                node_spacing=node_spacing,
                kernel_size=kernel_size,
                scale_factor=scale_factor,
                min_spur_length=min_spur_length,
            )

            with open(os.path.join(dirs["graphs"], f"{img_name}.p"), "wb") as f:
                pickle.dump(artifacts.graph_dict, f)

            cv2.imwrite(
                os.path.join(dirs["clean_masks"], f"{img_name}.png"),
                artifacts.clean_mask,
            )
            cv2.imwrite(
                os.path.join(dirs["images"], f"{img_name}.png"), artifacts.image
            )
            cv2.imwrite(
                os.path.join(dirs["overlays"], f"{img_name}.png"), artifacts.overlay
            )
            cv2.imwrite(
                os.path.join(dirs["keypoint_masks"], f"{img_name}.png"),
                artifacts.keypoint_mask,
            )

        except Exception as e:
            print(f"Failed to process {img_name}: {e}")

    print(f"Successfully generated dataset artifacts in {output_dir}")
