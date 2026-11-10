import bisect
from collections import Counter, defaultdict
import logging
import os
import pickle
import time

import numpy as np


def _eval(pair):
    pair.sort(key=lambda x: x[0])
    vid2cid = defaultdict(list)
    gt_size = defaultdict(int)
    cid_size = defaultdict(int)
    for i, j in pair:
        vid2cid[i].append(j)
        cid_size[j] += 1
        if i != -1:
            gt_size[i] += 1
    vid2cid.pop(-1, None)
    assert len(gt_size) == len(vid2cid)
    precision = 0
    recall = 0
    vid_to_cid = {}
    for vid, cids in vid2cid.items():
        cs = [i for i in cids if i != -1]
        if cs:
            cid, cnt = max(Counter(cs).items(), key=lambda x: x[1])
            precision += cnt / cid_size[cid] * gt_size[vid]
            recall += cnt
            vid_to_cid[vid] = cid
    gt_total = sum(gt_size.values())
    precision /= gt_total
    recall /= gt_total
    fscore = 2 * precision * recall / (precision + recall + 1e-8)
    expansion = sum(len(set(i)) for i in vid2cid.values()) / len(vid2cid)
    return precision, recall, fscore, expansion, vid_to_cid

def evaluate(records, labels, log=True, save_name=None):
    if not isinstance(labels, list):
        labels = labels.tolist()
    pair = [
        (-1 if i["vehicle_id"] is None else int(i["vehicle_id"]), j)
        for i, j in zip(records, labels)
    ]
    if save_name is not None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "eval_history")
        os.makedirs(path, exist_ok=True)
        name = f'{path}/{time.strftime("%Y%m%d_%H%M%S")}_{save_name}'
        pickle.dump(pair, open(name, "wb"))
        logging.info(f"saved to {name}")
    precision, recall, fscore, expansion, vid_to_cid = _eval(pair)
    if log:
        print("------------------")
        print(
            f"clusters: {len(set(labels)-set([-1]))}\nprecision: {precision}\nrecall:    {recall}\nfscore:    {fscore}\nexpansion: {expansion}"
        )
        print("------------------")
    return precision, recall, fscore, expansion, vid_to_cid


def gen_vid2cid(records, labels):
    if not isinstance(labels, list):
        labels = labels.tolist()
    pair = [
        (-1 if i["vehicle_id"] is None else int(i["vehicle_id"]), j)
        for i, j in zip(records, labels)
    ]
    vid2cid = defaultdict(set)
    for i, j in pair:
        vid2cid[i].add(j)
    vid2cid.pop(-1, None)
    return vid2cid

def eval_single_traj(ground_truth_range_set, path):
    """Node-indexed greedy spatio-temporal interval matching."""
    by_node = {}
    for item in ground_truth_range_set:
        by_node.setdefault(item[0], []).append(item)
    hit_num = 0
    for node_id, avg_time in path:
        cand = by_node.get(node_id)
        if cand:
            for idx, (_, start, end) in enumerate(cand):
                if start <= avg_time <= end:
                    hit_num += 1
                    cand.pop(idx)
                    break  # Consume at most one ground-truth interval per predicted point
    return hit_num

def eval_traj(
    records, 
    labels, 
    road_node_traj_gt_dict, 
    pred_traj_dict, 
    vid2cid=None,
    print_traj_res=False
):
    if vid2cid is None:
        vid2cid = gen_vid2cid(records, labels)
    
    precision = 0
    recall = 0
    vid_to_traj_id = {}
    for gt_label, traj_gt_list in road_node_traj_gt_dict.items():
        ground_truth_range_set = {
            (node_id, st, ed)
            for node_id, st, ed in traj_gt_list
        }
        gt_num = len(ground_truth_range_set)
        
        max_hit_num = 0
        final_pred_label = None
        for pred_label in vid2cid[gt_label]:
            if pred_label not in pred_traj_dict.keys():
                continue
            orig_traj = pred_traj_dict[pred_label]
            hit_num = eval_single_traj(ground_truth_range_set, orig_traj)
            if hit_num > max_hit_num:
                max_hit_num = hit_num
                final_pred_label = pred_label
        
        # calculate metrics
        path_len = len(pred_traj_dict[final_pred_label]) if final_pred_label is not None else 0
        precision += max_hit_num / path_len * gt_num if path_len else 0
        recall += max_hit_num
        vid_to_traj_id[gt_label] = final_pred_label
        
        curr_precision = max_hit_num / path_len if (final_pred_label is not None and path_len) else 0
        curr_recall = max_hit_num / gt_num if (final_pred_label is not None and gt_num) else 0
        curr_f1 = 2 * max_hit_num / (path_len + gt_num) if (path_len + gt_num) else 0
        if print_traj_res:
            print(f"GT/predicted {gt_label}/{final_pred_label} precision/recall/fscore: {curr_precision}/{curr_recall}/{curr_f1}")
    
    gt_total = sum([len(traj_gt_list) for traj_gt_list in road_node_traj_gt_dict.values()])
    precision = precision / gt_total if gt_total else 0.0
    recall = recall / gt_total if gt_total else 0.0
    fscore = 2 * precision * recall / (precision + recall + 1e-8)
    print("------------------")
    valid_traj_count = sum(1 for k in pred_traj_dict if int(k) >= 0)
    print(
        f"trajectories: {valid_traj_count}\nprecision: {precision}\nrecall:    {recall}\nfscore:    {fscore}"
    )
    print("------------------")
    return precision, recall, fscore, vid_to_traj_id


def eval_traj_macro(
    records,
    labels,
    road_node_traj_gt_dict,
    pred_traj_dict,
    vid2cid=None,
):
    """Compute unweighted per-vehicle Macro Precision, Macro Recall, and Macro F1."""
    if vid2cid is None:
        vid2cid = gen_vid2cid(records, labels)

    prec_list = []
    rec_list = []
    f1_list = []
    vid_to_traj_id = {}
    for gt_label, traj_gt_list in road_node_traj_gt_dict.items():
        ground_truth_range_set = {
            (node_id, st, ed)
            for node_id, st, ed in traj_gt_list
        }
        gt_num = len(ground_truth_range_set)
        max_hit_num = 0
        final_pred_label = None
        for pred_label in vid2cid[gt_label]:
            if pred_label not in pred_traj_dict:
                continue
            orig_traj = pred_traj_dict[pred_label]
            hit_num = eval_single_traj(ground_truth_range_set, orig_traj)
            if hit_num > max_hit_num:
                max_hit_num = hit_num
                final_pred_label = pred_label

        path_len = len(pred_traj_dict[final_pred_label]) if final_pred_label is not None else 0
        cp = max_hit_num / path_len if (final_pred_label is not None and path_len > 0) else 0.0
        cr = max_hit_num / gt_num if (final_pred_label is not None and gt_num > 0) else 0.0
        cf = (2.0 * max_hit_num / (path_len + gt_num)) if (path_len + gt_num) > 0 else 0.0
        prec_list.append(float(cp))
        rec_list.append(float(cr))
        f1_list.append(float(cf))
        vid_to_traj_id[gt_label] = final_pred_label

    macro_precision = float(np.mean(prec_list)) if prec_list else 0.0
    macro_recall = float(np.mean(rec_list)) if rec_list else 0.0
    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0
    return macro_precision, macro_recall, macro_f1, vid_to_traj_id


def eval_single_traj_stlc_interval(traj_gt, pred_traj, road_graph, lam=0.5):
    """Vectorized interval-based spatio-temporal linear combine (STLC) similarity."""
    nodes = road_graph.nodes
    gt_points = [
        (nodes[road_node]["x"], nodes[road_node]["y"], st, end)
        for road_node, st, end in traj_gt
        if "x" in nodes[road_node] and "y" in nodes[road_node]
    ]
    recovered_points = [
        (nodes[road_node]["x"], nodes[road_node]["y"], tm)
        for road_node, tm in pred_traj
        if "x" in nodes[road_node] and "y" in nodes[road_node]
    ]
    if not gt_points or not recovered_points:
        return 0.0

    gt_arr = np.asarray(gt_points, dtype=np.float64)
    rec_arr = np.asarray(recovered_points, dtype=np.float64)
    dx = gt_arr[:, 0:1] - rec_arr[:, 0][None, :]
    dy = gt_arr[:, 1:2] - rec_arr[:, 1][None, :]
    sdist = np.hypot(dx, dy)
    t_rec = rec_arr[:, 2][None, :]
    tdist = np.maximum(0.0, np.maximum(gt_arr[:, 2:3] - t_rec, t_rec - gt_arr[:, 3:4]))

    forward_time_sim = float(np.mean(np.exp(-np.min(tdist, axis=1))))
    backward_time_sim = float(np.mean(np.exp(-np.min(tdist, axis=0))))
    forward_space_sim = float(np.mean(np.exp(-np.min(sdist, axis=1))))
    backward_space_sim = float(np.mean(np.exp(-np.min(sdist, axis=0))))

    total_time_sim = forward_time_sim + backward_time_sim
    total_space_sim = forward_space_sim + backward_space_sim
    stic_score = lam * total_time_sim + (1.0 - lam) * total_space_sim
    return float(stic_score / 2.0)


def eval_traj_stlc(records, labels, road_node_traj_gt_dict, pred_traj_dict, road_graph, lam, vid2cid=None):
    if vid2cid is None:
        vid2cid = gen_vid2cid(records, labels)

    stic_score_list = []
    vid_to_traj_id = {}
    for gt_label, traj_gt_list in road_node_traj_gt_dict.items():
        max_stic_score = 0
        final_pred_label = None
        for pred_label in vid2cid[gt_label]:
            if pred_label not in pred_traj_dict.keys():
                continue
            pred_traj_list = pred_traj_dict[pred_label]
            stic_score = eval_single_traj_stlc_interval(traj_gt_list, pred_traj_list, road_graph, lam)
            if stic_score > max_stic_score:
                max_stic_score = stic_score
                final_pred_label = pred_label
        stic_score_list.append(max_stic_score)
        
        # assign the corresponding label
        vid_to_traj_id[gt_label] = final_pred_label
    
    final_stic_score = np.mean(stic_score_list) if stic_score_list else 0.0
    print("------------------")
    print(
        f"stlc score: {final_stic_score}"
    )
    print("------------------")
    return final_stic_score, vid_to_traj_id

def lcss(list1, list2):
    """
    Compute the Longest Common Subsequence (LCSS) length between two lists
    using the exact Hunt-Szymanski algorithm.
    """
    if not list1 or not list2:
        return 0
    pos = {}
    for idx, val in enumerate(list1):
        pos.setdefault(val, []).append(idx)
    for val in pos:
        pos[val].reverse()
    tails = []
    for val in list2:
        idxs = pos.get(val)
        if idxs:
            for i in idxs:
                k = bisect.bisect_left(tails, i)
                if k == len(tails):
                    tails.append(i)
                else:
                    tails[k] = i
    return len(tails)

def eval_traj_lcss(records, labels, road_node_traj_gt_dict, pred_traj_dict, vid2cid=None):
    if vid2cid is None:
        vid2cid = gen_vid2cid(records, labels)
    
    lcss_sim_list = []
    vid_to_traj_id = {}
    for gt_label, traj_gt_list in road_node_traj_gt_dict.items():
        traj_gt_node_list = [node_id for node_id, _, _ in traj_gt_list]
        max_lcss_sim = 0
        final_pred_label = None
        for pred_label in vid2cid[gt_label]:
            if pred_label not in pred_traj_dict.keys():
                continue
            pred_traj_list = pred_traj_dict[pred_label]
            pred_traj_node_list = [node_id for node_id, _ in pred_traj_list]
            lcss_len = lcss(traj_gt_node_list, pred_traj_node_list)
            lcss_sim = lcss_len / max(len(traj_gt_node_list), len(pred_traj_node_list))
            if lcss_sim > max_lcss_sim:
                max_lcss_sim = lcss_sim
                final_pred_label = pred_label
        lcss_sim_list.append(max_lcss_sim)
        
        # assign the corresponding label
        vid_to_traj_id[gt_label] = final_pred_label
    
    mean_lcss_sim = np.mean(lcss_sim_list)
    print("------------------")
    print(
        f"mean LCSS similarity: {mean_lcss_sim}"
    )
    print("------------------")
    return mean_lcss_sim, vid_to_traj_id

def calc_st_volume_sz_arr(traj_arr, scale_coef):
    """Compute spatio-temporal volume bounds and dimensions."""
    mins = np.min(traj_arr, axis=0)
    maxs = np.max(traj_arr, axis=0)
    y_min, x_min, t_min = mins[0], mins[1], mins[2]
    y_max, x_max, t_max = maxs[0], maxs[1], maxs[2]

    bounds = [y_min, y_max, x_min, x_max, t_min, t_max]
    y_dim = int(np.ceil((y_max - y_min) * scale_coef[0]))
    x_dim = int(np.ceil((x_max - x_min) * scale_coef[1]))
    t_dim = int(np.ceil((t_max - t_min) * scale_coef[2]))

    return bounds, (y_dim, x_dim, t_dim)

def any_point_in_bounds_arr(traj_arr, bounds):
    """Check whether any trajectory point falls within the specified bounds."""
    y_min, y_max, x_min, x_max, t_min, t_max = bounds
    y, x, t = traj_arr[:, 0], traj_arr[:, 1], traj_arr[:, 2]
    return np.any((y_min <= y) & (y <= y_max) &
                  (x_min <= x) & (x <= x_max) &
                  (t_min <= t) & (t <= t_max))

# === Trajectory interpolation (without max_points padding) ===
def rasterize_lines_batch_interp_with_id_np_nopad(start_idx, end_idx, batch_id):
    """
    NumPy batched interpolation and rounding to obtain voxels while preserving batch_id.
    Uses no max_points padding; each line segment is sampled according to its required number of points.
    """
    start_idx = start_idx.astype(np.float32)
    end_idx = end_idx.astype(np.float32)
    batch_id = batch_id.astype(np.int64)

    # Compute the number of steps per trajectory segment (maximum coordinate difference)
    dist = np.max(np.abs(end_idx - start_idx), axis=1)
    num_points_per_line = np.ceil(dist).astype(np.int64) + 1
    
    # Estimate the total number of points required
    total_points = np.sum(num_points_per_line)

    # Pre-allocate memory
    all_points = np.empty((total_points, 3), dtype=np.int64)
    all_bids = np.empty((total_points, 1), dtype=np.int64)

    idx = 0
    for i in range(len(start_idx)):
        steps = num_points_per_line[i]
        if steps <= 1:
            pts = start_idx[i:i+1]
        else:
            t = np.linspace(0, 1, steps, dtype=np.float32)[:, None]
            pts = start_idx[i] * (1 - t) + end_idx[i] * t
        pts = np.round(pts, 0, out=None).astype(np.int64, copy=False)
        
        # Fill pre-allocated arrays
        num_pts = pts.shape[0]
        all_points[idx:idx + num_pts] = pts
        all_bids[idx:idx + num_pts] = batch_id[i]
        idx += num_pts

    # Concatenate batch IDs and coordinates
    points_with_id = np.hstack((all_bids, all_points))
    points_with_id = np.unique(points_with_id, axis=0)  # Deduplicate
    return points_with_id

# === Batched trajectories -> voxel indices ===
def traj_to_voxel_indices_batch_np(trajs, scale_coef, bounds_min_list):
    """Convert batched trajectories to voxel indices (NumPy, interpolation without padding)."""
    n_traj = len(trajs)
    if n_traj == 0:
        return np.empty((0, 4), dtype=np.int64)

    # Count total line segments
    seg_counts = [max(len(traj) - 1, 0) for traj in trajs]
    total_seg  = int(np.sum(seg_counts))
    if total_seg == 0:
        return np.empty((0, 4), dtype=np.int64)

    start_idx = np.empty((total_seg, 3), dtype=np.float32)
    end_idx = np.empty((total_seg, 3), dtype=np.float32)
    batch_ids = np.empty((total_seg,),  dtype=np.int64)
    
    sp = 0
    for b, traj in enumerate(trajs):
        m = len(traj)
        if m < 2:
            continue
        vox = ((traj - bounds_min_list[b]) * scale_coef).astype(np.float32, copy=False)
        cnt = m - 1
        start_idx[sp:sp+cnt] = vox[:-1]
        end_idx[sp:sp+cnt]   = vox[1:]
        batch_ids[sp:sp+cnt] = b
        sp += cnt

    traj_voxels = rasterize_lines_batch_interp_with_id_np_nopad(start_idx, end_idx, batch_ids)
    return traj_voxels

# === Build sparse voxel sets ===
def _build_sparse_sets_np(vox_list):
    """Represent each trajectory as sorted unique voxel indices."""
    sets = []
    sizes = []
    for vox in vox_list:
        uniq = np.unique(vox, axis=0)
        sizes.append(len(uniq))
        sets.append(uniq)
    return sets, np.array(sizes, dtype=np.int64)

# === Compute set intersection count ===
def _intersection_count(a, b):
    """Exact row intersection count for sorted unique (N, 3) voxel arrays."""
    if a is None or b is None:
        return 0
    a_arr = np.ascontiguousarray(a, dtype=np.int64)
    b_arr = np.ascontiguousarray(b, dtype=np.int64)
    if a_arr.size == 0 or b_arr.size == 0:
        return 0
    dt = np.dtype((np.void, a_arr.dtype.itemsize * a_arr.shape[1]))
    return int(
        np.intersect1d(
            a_arr.view(dt).ravel(),
            b_arr.view(dt).ravel(),
            assume_unique=True,
        ).size
    )

# === Batched pairwise MSE matrix computation ===
def calc_pairwise_mse_set_batched(gt_vox_list, pred_vox_list, batch_size=32):
    """
    Set-intersection-based GT-Pred MSE matrix computation (NumPy CPU).
    """
    num_gt = len(gt_vox_list)
    num_pred = len(pred_vox_list)
    mse_matrix = np.zeros((num_pred, num_gt), dtype=np.float32)

    gt_sets, gt_sizes = _build_sparse_sets_np(gt_vox_list)
    pred_sets, pred_sizes = _build_sparse_sets_np(pred_vox_list)

    for g in range(num_gt):
        gset = gt_sets[g]
        gsize = max(gt_sizes[g], 1)
        for i in range(0, num_pred, batch_size):
            j = min(i + batch_size, num_pred)
            inter = np.fromiter(
                (_intersection_count(gset, pred_sets[k]) for k in range(i, j)),
                dtype=np.int64,
                count=j - i
            ).astype(np.float32, copy=False)
            # |A symmetric_diff B| = |A| + |B| - 2|A intersect B|
            preds = pred_sizes[i:j].astype(np.float32, copy=False)
            numerator = preds + gsize - 2.0 * inter
            denominator = preds + gsize
            mse_matrix[i:j, g] = numerator / denominator # Normalize by the sum of GT and predicted trajectory voxel counts
    return mse_matrix

# === Main volume MSE evaluation pipeline ===
def eval_traj_volume_mse_sparse(records, 
                                labels,
                                road_node_traj_gt_dict, 
                                pred_traj_dict,
                                road_graph, 
                                scale_coef_list, 
                                vid2cid=None, 
                                batch_size=16,
                                print_traj_res=False):
    """Pure NumPy CPU implementation with unpadded interpolation."""
    if vid2cid is None:
        vid2cid = gen_vid2cid(records, labels)

    mapp_vol_total_time, match_vol_cnt = 0, 0
    vol_mse_list, vid_to_traj_id = [], {}
    scale_coef_np = np.array(scale_coef_list, dtype=np.float32)
    # Coordinate cache
    node_coord_cache = {n: (road_graph.nodes[n]['y'], road_graph.nodes[n]['x'])
                        for n in road_graph.nodes}

    for gt_label, traj_gt_list in road_node_traj_gt_dict.items():
        if not traj_gt_list:
            continue

        # GT trajectory
        traj_gt_node_tm_array = np.array(
            [(node_coord_cache[node_id][0], node_coord_cache[node_id][1], (st + end) / 2)
             for node_id, st, end in traj_gt_list],
            dtype=np.float32
        )

        bounds, _ = calc_st_volume_sz_arr(traj_gt_node_tm_array, scale_coef_np)
        bounds_min = np.array([bounds[0], bounds[2], bounds[4]], dtype=np.float32)

        # GT voxel
        t0 = time.time()
        gt_voxels = traj_to_voxel_indices_batch_np(
            [traj_gt_node_tm_array], scale_coef_np, [bounds_min]
        )
        t1 = time.time()
        mapp_vol_total_time += (t1 - t0)
        
        # Collect candidate predicted trajectories
        pred_trajs_all, pred_labels_all = [], []
        for pred_label in vid2cid[gt_label]:
            pred_traj_list = pred_traj_dict.get(pred_label, None)
            if not pred_traj_list:
                continue
            pred_traj_node_arr = np.array(
                [(node_coord_cache[node_id][0], node_coord_cache[node_id][1], tm)
                 for node_id, tm in pred_traj_list],
                dtype=np.float32
            )
            if not any_point_in_bounds_arr(pred_traj_node_arr, bounds):
                continue
            pred_trajs_all.append(pred_traj_node_arr)
            pred_labels_all.append(pred_label)

        if not pred_trajs_all:
            vol_mse_list.append(1.0)
            vid_to_traj_id[gt_label] = None
            continue

        # Process predicted trajectories in batches
        min_vol_mse, final_pred_label = float('inf'), None
        for p_start in range(0, len(pred_trajs_all), batch_size):
            p_end = min(p_start + batch_size, len(pred_trajs_all))
            pred_trajs_batch = pred_trajs_all[p_start:p_end]
            pred_labels_batch = pred_labels_all[p_start:p_end]
            pred_bounds_min_list = [bounds_min] * len(pred_trajs_batch)

            # Pred voxel
            t0 = time.time()
            pred_vox_all = traj_to_voxel_indices_batch_np(
                pred_trajs_batch, scale_coef_np, pred_bounds_min_list
            )
            pred_vox_list = []
            for traj_id in range(len(pred_trajs_batch)):
                mask = pred_vox_all[:, 0] == traj_id
                pred_vox_list.append(pred_vox_all[mask, 1:])
            t1 = time.time()
            mapp_vol_total_time += (t1 - t0)

            # MSE
            mse_matrix = calc_pairwise_mse_set_batched(
                [gt_voxels[:, 1:]],
                pred_vox_list,
                batch_size=batch_size
            )
            match_vol_cnt += len(pred_vox_list)

            batch_min = mse_matrix.min()
            if batch_min < min_vol_mse:
                min_vol_mse = batch_min
                best_idx = mse_matrix.argmin()
                final_pred_label = pred_labels_batch[best_idx]

        if print_traj_res:
            print(f"GT/predicted {gt_label}/{final_pred_label} Vol-MSE: {min_vol_mse}")
        vol_mse_list.append(min_vol_mse)
        vid_to_traj_id[gt_label] = final_pred_label

    mean_vol_mse = np.mean(vol_mse_list)
    print(f'Mapping volume total time: {mapp_vol_total_time:.4f}s')
    print(f'Volume matching total times: {match_vol_cnt}')
    print("------------------")
    print(f"Mean volume MSE loss: {mean_vol_mse:.6f}")
    print("------------------")
    return mean_vol_mse, vid_to_traj_id, mapp_vol_total_time

def eval_trajs(
    records_all_sorted,
    pred_labels,
    road_node_traj_gt_dict,
    predicted_traj_dict,
    road_graph,
    scale_coef_list,
    vid2cid,
    print_mark,
    print_traj_res=False
):
    # 1. Precision/Recall/F1 (Micro & Macro)
    acc_time_start = time.time()
    precision, recall, fscore, _ = eval_traj(
        records_all_sorted, 
        pred_labels, 
        road_node_traj_gt_dict, 
        predicted_traj_dict, 
        vid2cid,
        print_traj_res
    )
    macro_precision, macro_recall, macro_f1, _ = eval_traj_macro(
        records_all_sorted,
        pred_labels,
        road_node_traj_gt_dict,
        predicted_traj_dict,
        vid2cid,
    )
    acc_time = time.time() - acc_time_start
    print(f'{print_mark} trajectory micro precision/recall/f1-score: {precision:.4f}/{recall:.4f}/{fscore:.4f}, macro precision/recall/f1-score: {macro_precision:.4f}/{macro_recall:.4f}/{macro_f1:.4f}, total consumes time: {acc_time:.2f}s')

    # 2. STLC
    stlc_time_start = time.time()
    final_stic_score, _ = eval_traj_stlc(
        records_all_sorted, 
        pred_labels, 
        road_node_traj_gt_dict, 
        predicted_traj_dict, 
        road_graph, 
        0.5, 
        vid2cid
    )
    stlc_time = time.time() - stlc_time_start
    print(f'{print_mark} trajectory mean STLC score: {final_stic_score:.4f}, total consumes time: {stlc_time:.2f}s')

    # 3. LCSS
    lcss_time_start = time.time()
    mean_lcss_sim, _ = eval_traj_lcss(
        records_all_sorted, 
        pred_labels, 
        road_node_traj_gt_dict, 
        predicted_traj_dict, 
        vid2cid
    )
    lcss_time = time.time() - lcss_time_start
    print(f'{print_mark} trajectory mean LCSS similarity: {mean_lcss_sim:.4f}, total consumes time: {lcss_time:.2f}s')

    # 4. Volume MSE
    vol_mse_time_start = time.time()
    mean_vol_mse, _, mapp_vol_time = eval_traj_volume_mse_sparse(
        records_all_sorted, 
        pred_labels, 
        road_node_traj_gt_dict, 
        predicted_traj_dict, 
        road_graph, 
        scale_coef_list, 
        vid2cid,
        print_traj_res=print_traj_res
    )
    vol_mse_time = time.time() - vol_mse_time_start
    print(f'{print_mark} trajectory mean volume MSE loss: {mean_vol_mse:.4f}, mapping consumes time: {mapp_vol_time:.2f}s, total consumes time: {vol_mse_time:.2f}s')

    # Return all evaluation metrics
    return {
        "precision": precision,
        "recall": recall,
        "fscore": fscore,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "stlc": final_stic_score,
        "lcss": mean_lcss_sim,
        "vol_mse": mean_vol_mse,
        "times": {
            "acc_time": acc_time,
            "stlc_time": stlc_time,
            "lcss_time": lcss_time,
            "mapping_time": mapp_vol_time,
            "vol_mse_time": vol_mse_time,
        }
    }
