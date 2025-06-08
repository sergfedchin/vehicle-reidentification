import os
import numpy as np
import json
import pickle
from tqdm import tqdm
import random
from datetime import datetime

def eval_func(distance_matrix: np.ndarray,
              query_vehicle_ids: np.ndarray,
              gallery_vehicle_ids: np.ndarray,
              query_camera_ids: np.ndarray,
              gallery_camera_ids: np.ndarray,
              max_rank: int = 50,
              remove_junk: bool = True,
              num_visual_samples: int = 0,
              output_dir: str = "vis_results"
              ) -> tuple[np.ndarray, float]:

    """
    Evaluation function that saves visualization data to files

    """
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"eval_{timestamp}")
    os.makedirs(output_path, exist_ok=True)
    
    # Compute metrics and collect samples
    n_query, n_gallery = distance_matrix.shape
    all_cmc, all_AP = [], []
    n_valid_query_images = 0
    sample_indices = random.sample(range(n_query), min(num_visual_samples, n_query)) if num_visual_samples > 0 else []
    
    for i in tqdm(range(n_query), desc='Evaluating', bar_format='{l_bar}{bar:20}{r_bar}', leave=True, unit='img'):
        q_id, q_cam = query_vehicle_ids[i], query_camera_ids[i]
        order = np.argsort(distance_matrix[i])
        
        if remove_junk:
            remove = (gallery_vehicle_ids[order] == q_id) & (gallery_camera_ids[order] == q_cam)
            keep = ~remove
        else:
            keep = np.ones_like(gallery_vehicle_ids, dtype=bool)
        
        # Save sample data if selected
        if i in sample_indices:
            sample_data = {
                'query_idx': i,
                'order': order,
                'scores': 1 - distance_matrix[i][order],  # Convert to similarity
                'matches': (gallery_vehicle_ids[order] == q_id).astype(int),
                'keep': keep
            }
            with open(os.path.join(output_path, f'sample_{i}.pkl'), 'wb') as f:
                pickle.dump(sample_data, f)
        
        # Metric calculation (same as before)
        orig_cmc = (gallery_vehicle_ids[order] == q_id).astype(int)[keep]
        if not orig_cmc.any():
            continue
            
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        n_valid_query_images += 1
        
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1) for i, x in enumerate(tmp_cmc)]
        AP = np.array(tmp_cmc)[orig_cmc.astype(bool)].sum() / num_rel
        all_AP.append(AP)
    
    # Save final metrics
    metrics = {
        'cmc': np.array(all_cmc).mean(axis=0).tolist(),
        'mAP': np.mean(all_AP)
    }
    with open(os.path.join(output_path, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)
    
    return metrics['cmc'], metrics['mAP']