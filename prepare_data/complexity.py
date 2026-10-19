import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import trimesh
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import entropy
from PIL import Image
import os
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def compute_vertex_areas(mesh):
    vertex_areas = np.zeros(len(mesh.vertices), dtype=np.float64)
    np.add.at(vertex_areas, mesh.faces.flatten(), np.repeat(mesh.area_faces, 3) / 3.0)
    return vertex_areas

def sample_rgb_texture_to_vertices(mesh, texture_path):
    try:
        img = Image.open(texture_path).convert("RGB")
        width, height = img.size
        img_data = np.asarray(img, dtype=np.float64) / 255.0
        uvs = mesh.visual.uv

        xs = np.clip((uvs[:, 0] * (width - 1)).astype(np.int64), 0, width - 1)
        ys = np.clip(((1.0 - uvs[:, 1]) * (height - 1)).astype(np.int64), 0, height - 1)
        return img_data[ys, xs]
    except Exception:
        return np.zeros((len(mesh.vertices), 3))

def compute_normal_entropy_1d(vertices, normals, areas, kdtree, radius, bin_edges):
    entropies = np.zeros(len(vertices), dtype=np.float64)
    neighbors_list = kdtree.query_ball_point(vertices, radius)
    nrm = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)

    for i, neighbors in enumerate(neighbors_list):
        if not neighbors: continue
        nb = np.array(neighbors, dtype=np.int64)
        w = areas[nb]
        n_nb = nrm[nb]

        m = (n_nb * w[:, None]).sum(axis=0)
        m_norm = np.linalg.norm(m)
        n_ref = nrm[i] if m_norm < 1e-12 else m / m_norm

        dots = np.clip(n_nb @ n_ref, -1.0, 1.0)
        s = (1.0 - dots) / 2.0

        hist, _ = np.histogram(s, bins=bin_edges, weights=w)
        total = hist.sum()
        if total > 0:
            probs = hist / total
            entropies[i] = entropy(probs[probs > 0], base=2)
    return entropies

def compute_tex_entropy(vertices, rgb_vals, areas, kdtree, radius, bin_edges):
    entropies = np.zeros(len(vertices), dtype=np.float64)
    neighbors_list = kdtree.query_ball_point(vertices, radius)
    sqrt3 = np.sqrt(3.0)

    for i, neighbors in enumerate(neighbors_list):
        if not neighbors: continue
        nb = np.array(neighbors, dtype=np.int64)
        w = areas[nb]
        t_nb = rgb_vals[nb]

        w_sum = w.sum()
        if w_sum < 1e-12: continue
        t_ref = (t_nb * w[:, None]).sum(axis=0) / w_sum

        s = np.linalg.norm(t_nb - t_ref, axis=1) / sqrt3
        s = np.clip(s, 0.0, 1.0)

        hist, _ = np.histogram(s, bins=bin_edges, weights=w)
        total = hist.sum()
        if total > 0:
            probs = hist / total
            entropies[i] = entropy(probs[probs > 0], base=2)
    return entropies

def bake_to_uv_map_fixed(mesh, values, original_tex_path, output_path):
    width, height = 1024, 1024
    if os.path.exists(original_tex_path):
        try:
            with Image.open(original_tex_path) as img: width, height = img.size
        except: pass

    dpi = 100
    fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    ax.tripcolor(mesh.visual.uv[:, 0], mesh.visual.uv[:, 1], mesh.faces,
                 values, cmap='gray', shading='gouraud', vmin=0, vmax=1)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)

def save_ply_jet_fixed(mesh, vals01, output_path):
    v = np.clip(vals01, 0.0, 1.0)
    cmap = plt.get_cmap("jet")
    colors = (cmap(v)[:, :3] * 255).astype(np.uint8)

    temp_mesh = mesh.copy()
    try:
        temp_mesh.visual = trimesh.visual.ColorVisuals(mesh=temp_mesh, vertex_colors=colors)
    except:
        temp_mesh.visual.vertex_colors = colors
    temp_mesh.export(output_path)

def process_subject(subject_path):
    try:
        obj_files = glob.glob(os.path.join(subject_path, "*.obj"))
        if not obj_files: return False, f"No OBJ: {subject_path}"
        obj_path = obj_files[0]
        tex_path = os.path.join(subject_path, "material0.jpeg")
        if not os.path.exists(tex_path): return False, f"No Tex: {subject_path}"

        mesh = trimesh.load(obj_path, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = list(mesh.geometry.values())[0]

        areas = compute_vertex_areas(mesh)
        kdtree = cKDTree(mesh.vertices)
        bbox_diag = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])

        radii = [bbox_diag * 0.02, bbox_diag * 0.01, bbox_diag * 0.005]
        bins_list = [30, 20, 10]

        tex_features = sample_rgb_texture_to_vertices(mesh, tex_path)
        normals = mesh.vertex_normals

        geo_norm_scales, tex_norm_scales = [], []

        ply_dir = os.path.join(subject_path, "ply")
        os.makedirs(ply_dir, exist_ok=True)

        for k, (r, b) in enumerate(zip(radii, bins_list)):
            edges = np.linspace(0.0, 1.0, b + 1)
            h_max = np.log2(b)

            h_geo = compute_normal_entropy_1d(mesh.vertices, normals, areas, kdtree, r, edges)
            geo_n = h_geo / (h_max + 1e-12)
            geo_norm_scales.append(geo_n)

            h_tex = compute_tex_entropy(mesh.vertices, tex_features, areas, kdtree, r, edges)
            tex_n = h_tex / (h_max + 1e-12)
            tex_norm_scales.append(tex_n)

            mx_n = np.maximum(geo_n, tex_n)

            save_ply_jet_fixed(mesh, geo_n, os.path.join(ply_dir, f"comp_geo_r{k}.ply"))
            save_ply_jet_fixed(mesh, tex_n, os.path.join(ply_dir, f"comp_tex_r{k}.ply"))
            save_ply_jet_fixed(mesh, mx_n, os.path.join(ply_dir, f"comp_max_r{k}.ply"))

        geo_ms = np.mean(np.stack(geo_norm_scales), axis=0)
        tex_ms = np.mean(np.stack(tex_norm_scales), axis=0)
        max_ms = np.maximum(geo_ms, tex_ms)

        save_ply_jet_fixed(mesh, max_ms, os.path.join(subject_path, "comp_max_mean.ply"))

        save_ply_jet_fixed(mesh, tex_ms, os.path.join(ply_dir, "comp_tex_mean.ply"))
        save_ply_jet_fixed(mesh, geo_ms, os.path.join(ply_dir, "comp_geo_mean.ply"))

        bake_to_uv_map_fixed(mesh, geo_ms, tex_path,
                             os.path.join(subject_path, "material0_complexity_geo.png"))
        bake_to_uv_map_fixed(mesh, tex_ms, tex_path,
                             os.path.join(subject_path, "material0_complexity_tex.png"))
        bake_to_uv_map_fixed(mesh, max_ms, tex_path, os.path.join(subject_path, "material0_complexity.png"))

        return True, subject_path
    except Exception as e:
        return False, f"Error {os.path.basename(subject_path)}: {str(e)}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", type=str, default="data/thuman2.0_raw")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--num_workers", type=int, default=min(16, os.cpu_count()))
    args = parser.parse_args()

    all_subject_paths = []
    for split in args.splits:
        split_path = os.path.join(args.raw_root, split)
        if os.path.exists(split_path):
            subs = [os.path.join(split_path, f) for f in os.listdir(split_path)
                    if os.path.isdir(os.path.join(split_path, f))]
            all_subject_paths.extend(sorted(subs))

    total_count = len(all_subject_paths)
    print(f"Processing {total_count} subjects with {args.num_workers} workers")

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_subject, path): path for path in all_subject_paths}
        pbar = tqdm(total=total_count)
        success_count = 0
        for future in as_completed(futures):
            success, result = future.result()
            if success:
                success_count += 1
            else:
                print(result)
            pbar.update(1)
        pbar.close()
    print(f"done: {success_count}/{total_count}")
