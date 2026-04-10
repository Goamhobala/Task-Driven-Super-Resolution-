import os
import glob
import cv2
import pickle
import numpy as np
import networkx as nx
from skimage.morphology import skeletonize
import sknw

def main():
    # Update this to where your original 256x256 satellite images are stored
    IMG_DIR = "/kaggle/input/datasets/sonisuyash/sentinel-2-roads-dataset/15765738/images_enhanced_png/images_enhanced_png"
    MASK_DIR = "/kaggle/input/datasets/sonisuyash/sentinel-2-roads-dataset/15765738/masks_png/masks_png"

    # The new root directory for the 1024x1024 dataset
    OUTPUT_ROOT_DIR = "/kaggle/working/InstaRoad/sam_road/sentinel2_test_1024"

    convert_dataset_to_graphs(
        img_dir=IMG_DIR,
        mask_dir=MASK_DIR,
        output_dir=OUTPUT_ROOT_DIR,
        node_spacing=5,        # Baseline distance in 256x256 space (auto-scales to 20px)
        kernel_size=10,         # Morphological closing kernel (applied directly to upscaled image)
        scale_factor=4.0,      # Upscale factor (4x turns 256x256 into 1024x1024)
        min_spur_length=3      # Removes dead ends shorter than 3 pixels in 256x256 space (auto-scales to 12px)
    )

def prune_graph_spurs(nx_graph, min_spur_length):
    """
    Removes short dead-end branches (spurs) from the graph that are
    often artifacts of skeletonizing jagged pixelated edges.
    """
    dead_ends = [n for n, d in nx_graph.degree() if d == 1]
    nodes_to_remove = []

    for n in dead_ends:
        neighbor = list(nx_graph.neighbors(n))[0]
        edge_data = nx_graph.get_edge_data(n, neighbor)

        if 'weight' in edge_data:
            length = edge_data['weight']
        else:
            length = len(edge_data.get('pts', []))

        if length < min_spur_length:
            nodes_to_remove.append(n)

    nx_graph.remove_nodes_from(nodes_to_remove)
    return nx_graph

def create_sam_road_graph(nx_graph, step_size):
    """
    Converts a networkx graph into the SAM-Road/CityScale dictionary format.
    Notice: scale_factor downscaling is removed. Coordinates remain in upscaled space.
    """
    sat2graph_dict = {}
    node_coords = {}

    for node_id, node_data in nx_graph.nodes(data=True):
        # sknw natively returns (row, col)
        r, c = int(node_data['o'][0]), int(node_data['o'][1])

        node_coords[node_id] = (r, c)
        if (r, c) not in sat2graph_dict:
            sat2graph_dict[(r, c)] = []

    for u, v, edge_data in nx_graph.edges(data=True):
        coord_u = node_coords[u]
        coord_v = node_coords[v]

        # The curve pixel coordinates (row, col)
        path = [(int(r), int(c)) for r, c in edge_data['pts']]

        # Flow direction check
        if len(path) > 0:
            dist_u_start = (coord_u[0] - path[0][0])**2 + (coord_u[1] - path[0][1])**2
            dist_u_end = (coord_u[0] - path[-1][0])**2 + (coord_u[1] - path[-1][1])**2
            if dist_u_end < dist_u_start:
                path.reverse()

        # Walk the path and sample points
        sampled_nodes = [coord_u]
        for pt in path:
            last_pt = sampled_nodes[-1]
            dist = np.sqrt((pt[0] - last_pt[0])**2 + (pt[1] - last_pt[1])**2)
            if dist >= step_size:
                sampled_nodes.append(pt)

        # Ensure the final node connects perfectly to the intersection 'v'
        if sampled_nodes[-1] != coord_v:
            sampled_nodes.append(coord_v)

        # Add bidirectional connections in (row, col)
        for i in range(len(sampled_nodes) - 1):
            n1 = sampled_nodes[i]
            n2 = sampled_nodes[i+1]

            if n1 not in sat2graph_dict: sat2graph_dict[n1] = []
            if n2 not in sat2graph_dict: sat2graph_dict[n2] = []

            if n2 not in sat2graph_dict[n1]: sat2graph_dict[n1].append(n2)
            if n1 not in sat2graph_dict[n2]: sat2graph_dict[n2].append(n1)

    return sat2graph_dict

def create_overlay(img, graph_dict):
    """Draws the final graph dictionary onto the RGB image for verification."""
    overlay = img.copy()

    # Draw edges (White lines)
    for r1, c1 in graph_dict:
        for r2, c2 in graph_dict[(r1, c1)]:
            # cv2 requires (x, y) which is (col, row)
            cv2.line(overlay, (c1, r1), (c2, r2), (255, 255, 255), 2)

    # Draw nodes (Green circles)
    for r, c in graph_dict:
        cv2.circle(overlay, (c, r), 4, (0, 255, 0), -1)

    return overlay

def process_single_tile(img_path, mask_path, node_spacing, kernel_size, scale_factor, min_spur_length):
    """Processes a single image/mask pair and returns all 4 upscaled artifacts."""

    # Load Image and Mask
    rgb = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 1. Upscale the mask and image
    if scale_factor > 1.0:
        h, w = binary_mask.shape
        new_size = (int(w * scale_factor), int(h * scale_factor))
        # Masks use Nearest Neighbor to stay binary
        binary_mask = cv2.resize(binary_mask, new_size, interpolation=cv2.INTER_NEAREST)
        # RGB uses Cubic for better visual upscaling
        upscaled_rgb = cv2.resize(rgb, new_size, interpolation=cv2.INTER_CUBIC)
    else:
        upscaled_rgb = rgb.copy()


    # Output the "clean lines" mask directly from the morphologically closed mask
    clean_line_mask = binary_mask.copy()

    # 2. Morphological Closing
    if kernel_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    # Output the "clean lines" mask directly from the morphologically closed mask
    clean_line_mask = binary_mask.copy()

    # 3. Skeletonize (Still required for sknw to extract the graph)
    skeleton = skeletonize(binary_mask > 0)

    # 4. Extract Graph
    graph = sknw.build_sknw(skeleton.astype(np.uint16), multi=False)

    # 5. Prune noisy spurs (scaled to the new resolution)
    if min_spur_length > 0:
        graph = prune_graph_spurs(graph, min_spur_length=(min_spur_length * scale_factor))

    # Create blank keypoint mask
    h, w = binary_mask.shape
    keypoint_mask = np.zeros((h, w), dtype=np.uint8)

    # Draw keypoints (degree != 2) with a scaled radius
    kp_radius = 3
    for node, degree in graph.degree():
        if degree != 2:
            r, c = int(graph.nodes[node]['o'][0]), int(graph.nodes[node]['o'][1])
            cv2.circle(keypoint_mask, (c, r), kp_radius, 255, -1)

    # 6. Convert to Dictionary format (node_spacing is scaled to the new resolution)
    sam_graph_dict = create_sam_road_graph(graph, step_size=(node_spacing * scale_factor))

    # 7. Generate Visual Overlay
    overlay_img = create_overlay(upscaled_rgb, sam_graph_dict)

    return sam_graph_dict, clean_line_mask, upscaled_rgb, overlay_img, keypoint_mask

def convert_dataset_to_graphs(img_dir, mask_dir, output_dir, node_spacing=5, kernel_size=3, scale_factor=2.0, min_spur_length=3):
    """
    Main function to process directories and save the 4 requested artifacts.
    """
    # Create output subdirectories
    dirs = {
        "graphs": os.path.join(output_dir, "graphs_p"),
        "clean_masks": os.path.join(output_dir, "clean_masks"),
        "images": os.path.join(output_dir, "images_1024"),
        "overlays": os.path.join(output_dir, "overlays"),
        "keypoint_masks": os.path.join(output_dir, "keypoint_masks")
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    mask_paths = glob.glob(os.path.join(mask_dir, "*.png"))
    print(f"Found {len(mask_paths)} masks to process in '{mask_dir}'.")

    for m_path in mask_paths:
        img_name = os.path.basename(m_path).split('.')[0]

        # Look for a matching RGB image (assumes it has the same base name and is a .png or .jpg)
        i_path_png = os.path.join(img_dir, f"{img_name}.png")
        i_path_jpg = os.path.join(img_dir, f"{img_name}.jpg")

        if os.path.exists(i_path_png):
            i_path = i_path_png
        elif os.path.exists(i_path_jpg):
            i_path = i_path_jpg
        else:
            print(f"Warning: Could not find matching original image for {img_name}. Skipping.")
            continue

        try:
            # Process the artifacts
            graph_dict, clean_mask, upscaled_img, overlay, keypoint_mask = process_single_tile(
                img_path=i_path,
                mask_path=m_path,
                node_spacing=node_spacing,
                kernel_size=kernel_size,
                scale_factor=scale_factor,
                min_spur_length=min_spur_length
            )

            # Save Graph Pickle
            with open(os.path.join(dirs["graphs"], f"{img_name}.p"), 'wb') as f:
                pickle.dump(graph_dict, f)

            # Save Clean Skeleton Mask
            cv2.imwrite(os.path.join(dirs["clean_masks"], f"{img_name}.png"), clean_mask)

            # Save Upscaled RGB
            cv2.imwrite(os.path.join(dirs["images"], f"{img_name}.png"), upscaled_img)

            # Save Debug Overlay
            cv2.imwrite(os.path.join(dirs["overlays"], f"{img_name}.png"), overlay)

            # Save Keypoint Mask
            cv2.imwrite(os.path.join(dirs["keypoint_masks"], f"{img_name}.png"), keypoint_mask)

        except Exception as e:
            print(f"Failed to process {img_name}: {e}")

    print(f"Successfully generated upscaled dataset artifacts in {output_dir}")


if __name__ == "__main__":
    # # Update this to where your original 256x256 satellite images are stored
    # IMG_DIR = "/Users/mistycloud/Projects/InstaRoad/data/sentinel2_test/images"
    # MASK_DIR = "/Users/mistycloud/Projects/InstaRoad/data/sentinel2_test/masks"

    # # The new root directory for the 1024x1024 dataset
    # OUTPUT_ROOT_DIR = "/Users/mistycloud/Projects/InstaRoad/data/sentinel2_test_1024"

    # convert_dataset_to_graphs(
    #     img_dir=IMG_DIR,
    #     mask_dir=MASK_DIR,
    #     output_dir=OUTPUT_ROOT_DIR,
    #     node_spacing=5,        # Baseline distance in 256x256 space (auto-scales to 20px)
    #     kernel_size=10,         # Morphological closing kernel (applied directly to upscaled image)
    #     scale_factor=4.0,      # Upscale factor (4x turns 256x256 into 1024x1024)
    #     min_spur_length=3      # Removes dead ends shorter than 3 pixels in 256x256 space (auto-scales to 12px)
    # )

    main()