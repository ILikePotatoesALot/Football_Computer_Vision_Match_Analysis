"""
YOLOv11 Football Tracking Module with Advanced Stats
Adapted for Streamlit + LLM integration
"""

from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np
import json
import os
import re
import torch
from torchreid import models


os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

# Load model once at module level (faster for multiple calls)
MODEL = None

def get_model():
    """Load the YOLO model"""
    global MODEL
    if MODEL is None:
        try:
            MODEL = YOLO("Football_Custom.pt")
        except:
             MODEL = YOLO("C:/Users/Austi/Sem4/Sem4 GAIT/GAIT ASSG/football-analyzer/Initia/Football_Custom.pt")
    return MODEL


def extract_reid_embedding(frame, bbox, reid_model, device='cuda'):
    """
    Extract ReID features from player crop
    
    Args:
        frame: Full video frame
        bbox: [x1, y1, x2, y2] player bounding box
        reid_model: Pretrained ReID model
        device: 'cuda' or 'cpu'
    
    Returns:
        512-dim embedding vector
    """
    try:
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure valid crop
        h, w = frame.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        # Extract crop
        crop = frame[y1:y2, x1:x2]
        
        if crop.size == 0:
            return None
        
        # Resize to ReID standard size (256x128)
        crop_resized = cv2.resize(crop, (128, 256))
        
        # Convert to tensor (RGB, normalized)
        crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
        crop_tensor = torch.from_numpy(crop_rgb).permute(2, 0, 1).float() / 255.0
        
        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        crop_tensor = (crop_tensor - mean) / std
        
        # Add batch dimension and move to device
        crop_tensor = crop_tensor.unsqueeze(0).to(device)
        
        # Extract features
        with torch.no_grad():
            embedding = reid_model(crop_tensor)
        
        # Return as numpy array
        return embedding.cpu().numpy().flatten()
        
    except Exception as e:
        print(f"ReID extraction error: {e}")
        return None


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def match_reid_embedding(new_embedding, existing_embeddings, threshold=0.7):
    """
    Find best match for new embedding among existing players
    
    Args:
        new_embedding: 512-dim vector for new detection
        existing_embeddings: dict {clean_id: embedding_vector}
        threshold: minimum similarity to consider a match
    
    Returns:
        (matched_id, similarity) or (None, 0.0) if no match
    """
    if new_embedding is None or len(existing_embeddings) == 0:
        return None, 0.0
    
    best_id = None
    best_similarity = 0.0
    
    for player_id, stored_emb in existing_embeddings.items():
        if stored_emb is None:
            continue
        
        similarity = cosine_similarity(new_embedding, stored_emb)
        
        if similarity > best_similarity and similarity > threshold:
            best_similarity = similarity
            best_id = player_id
    
    return best_id, best_similarity


def update_embedding_ema(old_embedding, new_embedding, alpha=0.3):
    """
    Update embedding using exponential moving average
    
    Args:
        old_embedding: Previous embedding
        new_embedding: New embedding
        alpha: Weight for new embedding (0.0 to 1.0)
    
    Returns:
        Updated embedding
    """
    if old_embedding is None:
        return new_embedding
    if new_embedding is None:
        return old_embedding
    
    return alpha * new_embedding + (1 - alpha) * old_embedding



def to_python_number(x):
    """Convert NumPy scalars/arrays to plain Python types for JSON."""
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.ndarray,)):
        return [to_python_number(v) for v in x.tolist()]
    return x

def parse_merge_instructions(context_text):
    """
    Examples:
    - Merge P#5, P#12 into P#5 (2 players)
    - Merge P#3, P#8, P#15 into P#3 (3 players)  
    - Merge P#12, P#26, P#30 into P#30 (3 players)
    - Merge P#1, P#2, P#3, P#4, P#5, P#6 into P#1 (6 players)

    Returns:
        dict: {target_id: [source_ids_to_merge]}
    """
    import re

    merge_map = {}

    if not context_text:
        return merge_map

    # Process each line separately
    lines = context_text.split('\n')

    for line in lines:
        line_lower = line.lower()

        # Must contain "merge" and "into"
        if 'merge' not in line_lower or 'into' not in line_lower:
            continue

        # Split by "into" to separate sources from target
        parts = re.split(r'\s+into\s+', line, flags=re.IGNORECASE)

        if len(parts) != 2:
            continue

        source_part, target_part = parts[0], parts[1]

        # Extract ALL P# numbers (case insensitive, supports commas)
        source_ids = [int(x) for x in re.findall(r'[Pp]#(\d+)', source_part)]
        target_ids = [int(x) for x in re.findall(r'[Pp]#(\d+)', target_part)]

        # Must have at least one target and sources
        if not target_ids or not source_ids:
            continue

        # Target = first P# after "into"
        target = target_ids[0]

        if target not in merge_map:
            merge_map[target] = []

        # Add ALL source players (unlimited!)
        merge_map[target].extend(source_ids)

    # Clean up: remove duplicates and target from its own source list
    for target in merge_map:
        merge_map[target] = list(set(s for s in merge_map[target] if s != target))

    return merge_map

def merge_player_stats(detailed_stats, merge_instructions):
    """
    Merge player statistics based on user instructions.

    Args:
        detailed_stats (dict): Player stats {player_id: stats_dict}
        merge_instructions (dict): {target_id: [source_ids]}

    Returns:
        tuple: (merged_stats, merge_log)
    """
    if not merge_instructions:
        return detailed_stats, []

    merged_stats = detailed_stats.copy()
    merge_log = []

    for target_id, source_ids in merge_instructions.items():
        if target_id not in merged_stats:
            merge_log.append(f"⚠️ Target P#{target_id} not found in stats")
            continue

        target_stats = merged_stats[target_id]

        for source_id in source_ids:
            if source_id not in merged_stats:
                merge_log.append(f"⚠️ Source P#{source_id} not found in stats")
                continue

            source_stats = merged_stats[source_id]

            # Sum all numeric fields
            target_stats['time_on_screen'] += source_stats['time_on_screen']
            target_stats['total_distance_pixels'] += source_stats['total_distance_pixels']
            target_stats['active_movement_frames'] += source_stats['active_movement_frames']
            target_stats['frames_detected'] += source_stats['frames_detected']
            target_stats['possession_frames'] += source_stats['possession_frames']
            target_stats['distance_with_ball_pixels'] += source_stats['distance_with_ball_pixels']
            target_stats['touches'] += source_stats['touches']
            target_stats['sprint_count'] += source_stats['sprint_count']
            target_stats['sprint_distance_pixels'] += source_stats['sprint_distance_pixels']
            target_stats['time_in_defensive_third'] += source_stats['time_in_defensive_third']
            target_stats['time_in_middle_third'] += source_stats['time_in_middle_third']
            target_stats['time_in_attacking_third'] += source_stats['time_in_attacking_third']

            # Recalculate percentages and averages
            total_frames = target_stats['frames_detected']
            if total_frames > 0:
                target_stats['activity_percentage'] = \
                    (target_stats['active_movement_frames'] / total_frames * 100)
                target_stats['possession_percentage'] = \
                    (target_stats['possession_frames'] / total_frames * 100)
                target_stats['avg_speed_pixels_per_frame'] = \
                    (target_stats['total_distance_pixels'] / max(total_frames - 1, 1))

            # Take max for top speed
            target_stats['top_speed_pixels_per_frame'] = max(
                target_stats['top_speed_pixels_per_frame'],
                source_stats['top_speed_pixels_per_frame']
            )

            # Take earliest appearance
            target_stats['first_appearance'] = min(
                target_stats['first_appearance'],
                source_stats['first_appearance']
            )

            merge_log.append(f"✅ Merged P#{source_id} → P#{target_id}")

            # Remove source player
            del merged_stats[source_id]

    return merged_stats, merge_log


def process_football_video(input_path, output_path, progress_callback=None):
    """
    Process football video with YOLOv8 tracking and advanced statistics

    Args:
        input_path (str): Path to input video file
        output_path (str): Path to save annotated output video
        progress_callback (callable, optional): Function to call with progress updates

    Returns:
        tuple: (output_video_path, detailed_stats_dict, llm_optimized_dict)
    """

    from sklearn.cluster import KMeans


    def extract_jersey_color(frame, bbox):
        """Extract dominant jersey color from player's torso region"""
        x1, y1, x2, y2 = map(int, bbox)

        height = y2 - y1
        width = x2 - x1

        torso_y1 = y1 + int(height * 0.1)
        torso_y2 = y1 + int(height * 0.5)
        torso_x1 = x1 + int(width * 0.2)
        torso_x2 = x2 - int(width * 0.2)

        torso_y1 = max(0, min(torso_y1, frame.shape[0] - 1))
        torso_y2 = max(0, min(torso_y2, frame.shape[0] - 1))
        torso_x1 = max(0, min(torso_x1, frame.shape[1] - 1))
        torso_x2 = max(0, min(torso_x2, frame.shape[1] - 1))

        if torso_y2 <= torso_y1 or torso_x2 <= torso_x1:
            return (128, 128, 128)

        torso = frame[torso_y1:torso_y2, torso_x1:torso_x2]

        if torso.size == 0:
            return (128, 128, 128)

        pixels = torso.reshape(-1, 3)
        mask = (pixels.mean(axis=1) > 30) & (pixels.mean(axis=1) < 240)
        filtered_pixels = pixels[mask]

        if len(filtered_pixels) < 10:
            filtered_pixels = pixels

        try:
            kmeans = KMeans(n_clusters=1, random_state=42, n_init=3)
            kmeans.fit(filtered_pixels)
            dominant_color = kmeans.cluster_centers_[0]
            return tuple(dominant_color.astype(int))
        except:
            return tuple(np.median(filtered_pixels, axis=0).astype(int))

    def assign_teams(player_colors, goalkeeper_ids):
        """Assign players to teams based on jersey colors"""
        if len(player_colors) < 2:
            return {pid: 'Team A' for pid in player_colors.keys()}

        field_players = {}
        for pid, color in player_colors.items():
            if pid not in goalkeeper_ids:
                field_players[pid] = color

        if len(field_players) < 4:
            team_assignment = {pid: 'Team A' for pid in field_players.keys()}
        else:
            player_ids = list(field_players.keys())
            colors = np.array([field_players[pid] for pid in player_ids])

            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            team_labels = kmeans.fit_predict(colors)

            team_assignment = {}
            for pid, label in zip(player_ids, team_labels):
                team_assignment[pid] = 'Team A' if label == 0 else 'Team B'

        for gk_id in goalkeeper_ids:
            if gk_id in player_colors:
                team_assignment[gk_id] = 'Goalkeeper'

        return team_assignment

    # Load model
    model = get_model()

    # Initialize ReID model
    print("Loading ReID model...")
    try:
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        reid_model = models.build_model(
            name='osnet_x1_0',
            num_classes=1000,
            pretrained=True
        )
        reid_model = reid_model.to(device)
        reid_model.eval()
        
        USE_REID = True
        REID_THRESHOLD = 0.6
        REID_UPDATE_ALPHA = 0.3  # EMA weight for embedding updates
        
        print(f"✅ ReID model loaded on {device}")
        
    except Exception as e:
        print(f"⚠️ ReID not available: {e}")
        print("   Continuing without ReID...")
        USE_REID = False
        reid_model = None
        device = 'cpu'

    # Constants
    BALL_CLASS_ID = 0
    GOALKEEPER_CLASS_ID = 1
    REFEREE_CLASS_ID = 3
    WARMUP_FRAMES = 30
    MIN_FRAMES_TO_COUNT = 30

    POSSESSION_RADIUS = 80
    WALKING_THRESHOLD = 3
    JOGGING_THRESHOLD = 8
    SPRINTING_THRESHOLD = 13
    MIN_SPRINT_FRAMES = 3

    # Get video info
    video_info = sv.VideoInfo.from_video_path(input_path)

    # Setup trackers
    player_tracker = sv.ByteTrack(
        track_activation_threshold=0.20,
        lost_track_buffer=90,
        minimum_matching_threshold=0.60,
        minimum_consecutive_frames=3,
        frame_rate=video_info.fps
    )

    ball_tracker = sv.ByteTrack(
        track_activation_threshold=0.001,
        lost_track_buffer=15,
        minimum_matching_threshold=0.99,
        minimum_consecutive_frames=1,
        frame_rate=video_info.fps
    )

    # Setup annotators (thin and clean)
    box_annotator = sv.BoxAnnotator(thickness=1)
    label_annotator = sv.LabelAnnotator(
        text_thickness=1,
        text_scale=0.4,
        text_padding=2
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=1,
        trace_length=20,
        position=sv.Position.BOTTOM_CENTER
    )

    # Storage
    player_stats = {}
    id_remap = {}
    next_clean_id = 1
    id_persistence = {}
    player_jersey_colors = {}
    goalkeeper_ids = set()
    player_embeddings = {} 
    reid_match_count = 0   

    # Setup VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(
        str(output_path),
        fourcc,
        video_info.fps,
        (video_info.width, video_info.height)
    )

    # Process video
    frame_generator = sv.get_video_frames_generator(input_path)

    for frame_idx, frame in enumerate(frame_generator):
        if progress_callback and frame_idx % 30 == 0:
            progress_callback(frame_idx, video_info.total_frames,
                            f"Processing frame {frame_idx}/{video_info.total_frames}")

        results = model(frame, verbose=False, conf=0.001, iou=0.3)[0]  #IMGSZ down to 640
        detections = sv.Detections.from_ultralytics(results)

        ball_mask = detections.class_id == BALL_CLASS_ID
        referee_mask = detections.class_id == REFEREE_CLASS_ID
        ball_detections = detections[ball_mask]
        referee_detections = detections[referee_mask]
        player_detections = detections[~ball_mask & ~referee_mask]


        gk_mask = player_detections.class_id == GOALKEEPER_CLASS_ID
        gk_detections = player_detections[gk_mask]
        other_players = player_detections[~gk_mask]
        
        #  confidence threshold
        gk_detections = gk_detections[gk_detections.confidence > 0.20]
        
        if len(gk_detections) > 0:
            gk_centers_x = (gk_detections.xyxy[:, 0] + gk_detections.xyxy[:, 2]) / 2
            field_center_x = video_info.width / 2
            
            #  Position filtering (outer 25%)
            left_goal_zone = video_info.width * 0.25
            right_goal_zone = video_info.width * 0.75
            valid_gk_mask = (gk_centers_x < left_goal_zone) | (gk_centers_x > right_goal_zone)
            gk_detections = gk_detections[valid_gk_mask]
            
            if len(gk_detections) > 0:
                # Recalculate centers after filtering
                gk_centers_x = (gk_detections.xyxy[:, 0] + gk_detections.xyxy[:, 2]) / 2
                
                left_gks = gk_centers_x < field_center_x
                right_gks = ~left_gks
                
                selected_indices = []
                
                if left_gks.any():
                    left_indices = np.where(left_gks)[0]
                    best_left = left_indices[np.argmax(gk_detections.confidence[left_indices])]
                    selected_indices.append(best_left)
                
                if right_gks.any():
                    right_indices = np.where(right_gks)[0]
                    best_right = right_indices[np.argmax(gk_detections.confidence[right_indices])]
                    selected_indices.append(best_right)
                
                if selected_indices:
                    gk_detections = sv.Detections(
                        xyxy=gk_detections.xyxy[selected_indices],
                        confidence=gk_detections.confidence[selected_indices],
                        class_id=gk_detections.class_id[selected_indices]
                    )
                else:
                    gk_detections = sv.Detections.empty()
            else:
                gk_detections = sv.Detections.empty()
        else:
            gk_detections = sv.Detections.empty()
        
        # Combine GKs with field players
        if len(other_players) > 0 and len(gk_detections) > 0:
            player_detections = sv.Detections(
                xyxy=np.vstack([other_players.xyxy, gk_detections.xyxy]),
                confidence=np.concatenate([other_players.confidence, gk_detections.confidence]),
                class_id=np.concatenate([other_players.class_id, gk_detections.class_id])
            )
        elif len(gk_detections) > 0:
            player_detections = gk_detections
        else:
            player_detections = other_players


        player_detections = player_detections[player_detections.confidence > 0.08]
        ball_detections = ball_detections[ball_detections.confidence > 0.17]

        if len(ball_detections) > 0:
            max_conf_idx = np.argmax(ball_detections.confidence)
            ball_detections = sv.Detections(
                xyxy=ball_detections.xyxy[max_conf_idx:max_conf_idx+1],
                confidence=ball_detections.confidence[max_conf_idx:max_conf_idx+1],
                class_id=ball_detections.class_id[max_conf_idx:max_conf_idx+1]
            )

        if len(player_detections) > 0:
            areas = (player_detections.xyxy[:, 2] - player_detections.xyxy[:, 0]) * \
                    (player_detections.xyxy[:, 3] - player_detections.xyxy[:, 1])
            player_detections = player_detections[areas > 200]

        player_detections = player_tracker.update_with_detections(player_detections)

        player_detections_list = []
        player_labels = []

                # Process ball
        #Extract position for possession tracking ===
        ball_shown = False
        ball_position = None

        if len(ball_detections) > 0 and frame_idx >= WARMUP_FRAMES:
            best_ball_bbox = ball_detections.xyxy[0]
            best_ball_conf = ball_detections.confidence[0]
            
            if best_ball_conf > 0.02:
                ball_center_x = (best_ball_bbox[0] + best_ball_bbox[2]) / 2
                ball_center_y = (best_ball_bbox[1] + best_ball_bbox[3]) / 2
                ball_position = (ball_center_x, ball_center_y)
                
                player_detections_list.append({
                    'bbox': best_ball_bbox,
                    'id': 999,
                    'confidence': best_ball_conf
                })
                player_labels.append(f"!BALL ({best_ball_conf:.2f})")
                ball_shown = True


       


        # Process referees
        if len(referee_detections) > 0 and frame_idx >= WARMUP_FRAMES:
            referee_detections = referee_detections[referee_detections.confidence > 0.3]
            referee_detections = ball_tracker.update_with_detections(referee_detections)

            for tracker_id, bbox, confidence in zip(
                referee_detections.tracker_id,
                referee_detections.xyxy,
                referee_detections.confidence
            ):
                referee_display_id = 900 + tracker_id

                player_detections_list.append({
                    'bbox': bbox,
                    'id': referee_display_id,
                    'confidence': confidence
                })
                player_labels.append(f"REF #{tracker_id}")

        # Find closest player to ball
        closest_player_clean_id = None
        closest_distance = float('inf')

        if ball_position is not None and frame_idx >= WARMUP_FRAMES:
            for tracker_id, bbox, confidence, class_id in zip(
                player_detections.tracker_id,
                player_detections.xyxy,
                player_detections.confidence,
                player_detections.class_id
            ):
                if class_id == BALL_CLASS_ID:
                    continue

                if tracker_id not in id_persistence:
                    id_persistence[tracker_id] = 0
                id_persistence[tracker_id] += 1

                if id_persistence[tracker_id] < 3:
                    continue

                                # Check persistence
                if tracker_id not in id_persistence:
                    id_persistence[tracker_id] = 0
                id_persistence[tracker_id] += 1
                
                if id_persistence[tracker_id] < 3:
                    continue
                
                # : ReID-based ID assignment with safety checks
                clean_id = None
                reid_matched = False

                if USE_REID and reid_model is not None:
                    # Extract ReID embedding for this detection
                    current_embedding = extract_reid_embedding(frame, bbox, reid_model, device)
                    
                    if current_embedding is not None:
                        #  Check if this tracker_id already has a clean_id
                        if tracker_id in id_remap:
                            # ByteTrack is still tracking this player - keep their existing ID
                            clean_id = id_remap[tracker_id]
                            
                            # Update embedding
                            if clean_id in player_embeddings:
                                player_embeddings[clean_id] = update_embedding_ema(
                                    player_embeddings[clean_id],
                                    current_embedding,
                                    alpha=REID_UPDATE_ALPHA
                                )
                        else:
                            matched_id, similarity = match_reid_embedding(
                                current_embedding, 
                                player_embeddings,
                                threshold=REID_THRESHOLD
                            )
                            
                            #  Additional validation checks
                            if matched_id is not None:
                                # Check 1: Make sure we're not matching too many to same ID
                                if matched_id in id_remap.values():
                                    current_trackers_for_id = [tid for tid, cid in id_remap.items() if cid == matched_id]
                                    if len(current_trackers_for_id) > 0:
                                        # Someone else currently has this ID - require higher similarity
                                        if similarity < 0.85:
                                            matched_id = None  # Don't match, create new ID
                                
                                #  Verify spatial consistency 
                                if matched_id is not None and matched_id in player_stats:
                                    if len(player_stats[matched_id]['positions']) > 0:
                                        last_pos = player_stats[matched_id]['positions'][-1]
                                        current_pos = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                                        distance = np.sqrt(
                                            (current_pos[0] - last_pos[0])**2 +
                                            (current_pos[1] - last_pos[1])**2
                                        )
                                        # If player "teleported" more than 300 pixels, don't match
                                        if distance > 300:
                                            print(f"⚠️ ReID match rejected: distance {distance:.0f}px too large")
                                            matched_id = None
                            
                            if matched_id is not None:
                                # Valid ReID match
                                clean_id = matched_id
                                reid_matched = True
                                reid_match_count += 1
                                
                                # Update embedding with EMA
                                player_embeddings[clean_id] = update_embedding_ema(
                                    player_embeddings[clean_id],
                                    current_embedding,
                                    alpha=REID_UPDATE_ALPHA
                                )
                                
                                # Update id_remap
                                id_remap[tracker_id] = clean_id
                                
                                if frame_idx % 30 == 0:
                                    print(f"🔗 ReID: Matched tracker_{tracker_id} → player #{clean_id} (sim: {similarity:.3f})")

                # Fallback: Use existing logic if ReID didn't match
                if clean_id is None:
                    if tracker_id in id_remap:
                        clean_id = id_remap[tracker_id]
                    else:
                        # New player
                        clean_id = next_clean_id
                        id_remap[tracker_id] = clean_id
                        next_clean_id += 1
                        
                        # Store initial embedding if using ReID
                        if USE_REID and reid_model is not None:
                            current_embedding = extract_reid_embedding(frame, bbox, reid_model, device)
                            if current_embedding is not None:
                                player_embeddings[clean_id] = current_embedding
                                print(f"✨ New player #{clean_id} with ReID embedding")



                player_center_x = (bbox[0] + bbox[2]) / 2
                player_center_y = (bbox[1] + bbox[3]) / 2

                distance = np.sqrt(
                    (player_center_x - ball_position[0])**2 +
                    (player_center_y - ball_position[1])**2
                )

                if distance < closest_distance:
                    closest_distance = distance
                    closest_player_clean_id = clean_id

        # Process players
        for tracker_id, bbox, confidence, class_id in zip(
            player_detections.tracker_id,
            player_detections.xyxy,
            player_detections.confidence,
            player_detections.class_id
        ):
            # Skip during warmup
            if frame_idx < WARMUP_FRAMES:
                continue

            if class_id == BALL_CLASS_ID:
                continue

            if tracker_id not in id_persistence:
                id_persistence[tracker_id] = 0

            id_persistence[tracker_id] += 1

            if id_persistence[tracker_id] < 3:
                continue

            if tracker_id not in id_remap:
                id_remap[tracker_id] = next_clean_id
                next_clean_id += 1

            clean_id = id_remap[tracker_id]

            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2

            if clean_id not in player_stats:
                player_stats[clean_id] = {
                    'frames_detected': 0,
                    'positions': [],
                    'first_seen_frame': frame_idx,
                    'possession_frames': 0,
                    'last_had_possession': False,
                    'touches': 0,
                    'distance_with_ball': 0.0,
                    'possession_sequences': [],
                    'current_possession_start': None,
                    'speeds': [],
                    'sprint_frames': 0,
                    'sprint_distance': 0.0,
                    'walking_frames': 0,
                    'jogging_frames': 0,
                    'sprinting_frames': 0,
                    'in_sprint': False,
                    'current_sprint_frames': 0,
                    'zone_frames': {'defensive': 0, 'middle': 0, 'attacking': 0}
                }

            # Only closest player gets possession
            has_possession = False
            if ball_position is not None:
                if clean_id == closest_player_clean_id and closest_distance < POSSESSION_RADIUS:
                    has_possession = True
                else:
                    has_possession = False

            if has_possession:
                player_stats[clean_id]['possession_frames'] += 1

                if not player_stats[clean_id]['last_had_possession']:
                    player_stats[clean_id]['touches'] += 1
                    player_stats[clean_id]['current_possession_start'] = frame_idx

                if len(player_stats[clean_id]['positions']) > 0:
                    prev_pos = player_stats[clean_id]['positions'][-1]
                    dist = np.sqrt((center_x - prev_pos[0])**2 + (center_y - prev_pos[1])**2)
                    player_stats[clean_id]['distance_with_ball'] += dist
            else:
                if player_stats[clean_id]['last_had_possession']:
                    if player_stats[clean_id]['current_possession_start'] is not None:
                        duration = frame_idx - player_stats[clean_id]['current_possession_start']
                        player_stats[clean_id]['possession_sequences'].append(duration)

            player_stats[clean_id]['last_had_possession'] = has_possession

            # Pace tracking
            if len(player_stats[clean_id]['positions']) > 0:
                prev_pos = player_stats[clean_id]['positions'][-1]
                speed = np.sqrt((center_x - prev_pos[0])**2 + (center_y - prev_pos[1])**2)
                player_stats[clean_id]['speeds'].append(speed)

                if speed < WALKING_THRESHOLD:
                    player_stats[clean_id]['walking_frames'] += 1
                    player_stats[clean_id]['in_sprint'] = False
                    player_stats[clean_id]['current_sprint_frames'] = 0
                elif speed < JOGGING_THRESHOLD:
                    player_stats[clean_id]['jogging_frames'] += 1
                    player_stats[clean_id]['in_sprint'] = False
                    player_stats[clean_id]['current_sprint_frames'] = 0
                else:
                    player_stats[clean_id]['sprinting_frames'] += 1
                    player_stats[clean_id]['sprint_distance'] += speed
                    player_stats[clean_id]['current_sprint_frames'] += 1

                    if not player_stats[clean_id]['in_sprint'] and \
                       player_stats[clean_id]['current_sprint_frames'] >= MIN_SPRINT_FRAMES:
                        player_stats[clean_id]['sprint_frames'] += 1
                        player_stats[clean_id]['in_sprint'] = True

            # Zone tracking
            third_width = video_info.width / 3
            if center_x < third_width:
                player_stats[clean_id]['zone_frames']['defensive'] += 1
            elif center_x < 2 * third_width:
                player_stats[clean_id]['zone_frames']['middle'] += 1
            else:
                player_stats[clean_id]['zone_frames']['attacking'] += 1

            player_stats[clean_id]['frames_detected'] += 1
            player_stats[clean_id]['positions'].append((center_x, center_y))

            player_detections_list.append({
                'bbox': bbox,
                'id': clean_id,
                'confidence': confidence
            })

            team = player_stats[clean_id].get('team', 'Team?')

            if class_id == GOALKEEPER_CLASS_ID:
                goalkeeper_ids.add(clean_id)
                player_labels.append(f"??? GK #{clean_id}")
            elif team == 'Goalkeeper':
                player_labels.append(f"??? GK #{clean_id}")
            elif team in ['Team A', 'Team B']:
                player_labels.append(f"{team} #{clean_id}")
            else:
                player_labels.append(f"P#{clean_id}")

            if frame_idx % 10 == 0:
                jersey_color = extract_jersey_color(frame, bbox)
                if clean_id not in player_jersey_colors:
                    player_jersey_colors[clean_id] = []
                player_jersey_colors[clean_id].append(jersey_color)

                if frame_idx % 30 == 0 and len(player_jersey_colors) > 2:
                    temp_avg_colors = {}
                    for pid, colors in player_jersey_colors.items():
                        if len(colors) > 0:
                            temp_avg_colors[pid] = tuple(np.mean(colors, axis=0).astype(int))

                    team_assignments = assign_teams(temp_avg_colors, goalkeeper_ids)

                if frame_idx % 60 == 0:  # Check every 60 frames (2 seconds)
                    # Define goal zones (outer 15% of field width on each side)
                    field_left_boundary = video_info.width * 0.15
                    field_right_boundary = video_info.width * 0.85
                    
              
                    for gk_id in list(goalkeeper_ids):  
                        if gk_id in player_stats and len(player_stats[gk_id]['positions']) > 30:
                            recent_positions = player_stats[gk_id]['positions'][-30:]
                            recent_x_positions = [pos[0] for pos in recent_positions]
                            
                            # Count how many frames the "goalkeeper" was in the middle of the field
                            out_of_zone_count = sum(
                                1 for x in recent_x_positions 
                                if field_left_boundary <= x <= field_right_boundary
                            )
                            
                            # If player spends >80% of time in midfield, they're NOT a goalkeeper
                            out_of_zone_ratio = out_of_zone_count / len(recent_x_positions)
                            
                            if out_of_zone_ratio > 0.80:
                                goalkeeper_ids.discard(gk_id)
                                print(f"⚠️ Player #{gk_id} removed from goalkeepers (in midfield {out_of_zone_ratio*100:.0f}% of time)")

                    for pid, team in team_assignments.items():
                        if pid in player_stats:
                            player_stats[pid]['team'] = team

        # Annotate frame
        if len(player_detections_list) > 0:
            player_bboxes = np.array([p['bbox'] for p in player_detections_list])
            player_ids = np.array([p['id'] for p in player_detections_list])
            player_confidences = np.array([p['confidence'] for p in player_detections_list])

            filtered_detections = sv.Detections(
                xyxy=player_bboxes,
                confidence=player_confidences,
                class_id=np.zeros(len(player_bboxes), dtype=int),
                tracker_id=player_ids
            )

            annotated_frame = box_annotator.annotate(
                scene=frame.copy(),
                detections=filtered_detections
            )
            annotated_frame = trace_annotator.annotate(
                scene=annotated_frame,
                detections=filtered_detections
            )
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame,
                detections=filtered_detections,
                labels=player_labels
            )
        else:
            annotated_frame = frame.copy()

        # Frame overlay
        if frame_idx < WARMUP_FRAMES:
            overlay_text = f"WARMING UP... {frame_idx}/{WARMUP_FRAMES}"
            color = (0, 165, 255)
            cv2.putText(
                annotated_frame,
                overlay_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )
            out.write(annotated_frame)
            continue
        else:
            ball_indicator = "B" if ball_shown else "NB"
            overlay_text = f"Frame: {frame_idx} | Active: {len(player_detections_list)} {ball_indicator}"
            color = (0, 255, 0)
            cv2.putText(
                annotated_frame,
                overlay_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )
            out.write(annotated_frame)

    out.release()
    if USE_REID:
        print(f"\n📊 ReID Statistics:")
        print(f"   Total ReID matches: {reid_match_count}")
        print(f"   Unique players tracked: {len(player_embeddings)}")
        print(f"   Average embeddings per player: {reid_match_count / max(len(player_embeddings), 1):.1f}")

    # Merge split IDs
    print("\n🔗 Checking for split player IDs...")

    # Cleanup
    cleaned_stats = {
        pid: stats for pid, stats in player_stats.items()
        if stats['frames_detected'] >= MIN_FRAMES_TO_COUNT
    }

    # Calculate final stats
    detailed_stats = {}
    sorted_cleaned = sorted(cleaned_stats.items(), key=lambda x: x[1]['first_seen_frame'])


    player_avg_colors = {}
    for pid, colors in player_jersey_colors.items():
        if len(colors) > 0:
            player_avg_colors[pid] = tuple(np.mean(colors, axis=0).astype(int))

    team_assignments = assign_teams(player_avg_colors, goalkeeper_ids)

    for player_id, data in cleaned_stats.items():
        positions = np.array(data['positions'])

        if len(positions) < 5:
            continue

        distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
        total_distance_pixels = np.sum(distances)
        avg_speed_pixels = np.mean(distances) if len(distances) > 0 else 0
        time_on_screen = data['frames_detected'] / video_info.fps

        movement_threshold = 5
        active_frames = np.sum(distances > movement_threshold) if len(distances) > 0 else 0

        possession_pct = (data['possession_frames'] / data['frames_detected'] * 100) if data['frames_detected'] > 0 else 0
        avg_possession_duration = (np.mean(data['possession_sequences']) / video_info.fps) if len(data['possession_sequences']) > 0 else 0

        top_speed = max(data['speeds']) if len(data['speeds']) > 0 else 0
        total_movement_frames = data['walking_frames'] + data['jogging_frames'] + data['sprinting_frames']
        walking_pct = (data['walking_frames'] / total_movement_frames * 100) if total_movement_frames > 0 else 0
        jogging_pct = (data['jogging_frames'] / total_movement_frames * 100) if total_movement_frames > 0 else 0
        sprinting_pct = (data['sprinting_frames'] / total_movement_frames * 100) if total_movement_frames > 0 else 0

        time_defensive = (data['zone_frames']['defensive'] / video_info.fps) if data['zone_frames']['defensive'] > 0 else 0
        time_middle = (data['zone_frames']['middle'] / video_info.fps) if data['zone_frames']['middle'] > 0 else 0
        time_attacking = (data['zone_frames']['attacking'] / video_info.fps) if data['zone_frames']['attacking'] > 0 else 0

        avg_position = np.mean(positions, axis=0).tolist()

        detailed_stats[player_id] = {
            'team': str(team_assignments.get(player_id, 'Unknown')),
            'jersey_color_rgb': to_python_number(player_avg_colors.get(player_id, (128, 128, 128))),
            'time_on_screen': float(time_on_screen),
            'total_distance_pixels': float(total_distance_pixels),
            'avg_speed_pixels_per_frame': float(avg_speed_pixels),
            'active_movement_frames': int(active_frames),
            'activity_percentage': float((active_frames / len(distances)) * 100) if len(distances) > 0 else 0.0,
            'first_appearance': int(data['first_seen_frame']),
            'frames_detected': int(data['frames_detected']),
            'possession_frames': int(data['possession_frames']),
            'possession_percentage': float(possession_pct),
            'distance_with_ball_pixels': float(data['distance_with_ball']),
            'touches': int(data['touches']),
            'avg_possession_duration_seconds': float(avg_possession_duration),
            'top_speed_pixels_per_frame': float(top_speed),
            'sprint_count': int(data['sprint_frames']),
            'sprint_distance_pixels': float(data['sprint_distance']),
            'walking_percentage': float(walking_pct),
            'jogging_percentage': float(jogging_pct),
            'sprinting_percentage': float(sprinting_pct),
            'time_in_defensive_third': float(time_defensive),
            'time_in_middle_third': float(time_middle),
            'time_in_attacking_third': float(time_attacking),
            'average_position': to_python_number(avg_position)
        }

    # Calculate team totals
    match_duration = video_info.total_frames / video_info.fps

    team_a_stats = {
        'total_possession_pct': 0,
        'total_distance': 0,
        'total_sprints': 0,
        'player_count': 0
    }
    team_b_stats = {
        'total_possession_pct': 0,
        'total_distance': 0,
        'total_sprints': 0,
        'player_count': 0
    }

    for pid, stats in detailed_stats.items():
        if stats['team'] == 'Team A':
            team_a_stats['total_possession_pct'] += stats['possession_percentage']
            team_a_stats['total_distance'] += stats['total_distance_pixels']
            team_a_stats['total_sprints'] += stats['sprint_count']
            team_a_stats['player_count'] += 1
        elif stats['team'] == 'Team B':
            team_b_stats['total_possession_pct'] += stats['possession_percentage']
            team_b_stats['total_distance'] += stats['total_distance_pixels']
            team_b_stats['total_sprints'] += stats['sprint_count']
            team_b_stats['player_count'] += 1

    # Top performers
    sorted_by_possession = sorted(detailed_stats.items(), key=lambda x: x[1]['possession_percentage'], reverse=True)
    sorted_by_distance = sorted(detailed_stats.items(), key=lambda x: x[1]['total_distance_pixels'], reverse=True)
    sorted_by_sprints = sorted(detailed_stats.items(), key=lambda x: x[1]['sprint_count'], reverse=True)

    llm_package = {
        "match_summary": {
            "duration_seconds": round(match_duration, 1),
            "total_players_tracked": len(detailed_stats),
            "team_a_players": team_a_stats['player_count'],
            "team_b_players": team_b_stats['player_count']
        },
        "team_totals": {
            "team_a": {
                "avg_possession_per_player": round(team_a_stats['total_possession_pct'] / max(team_a_stats['player_count'], 1), 1),
                "total_distance": int(team_a_stats['total_distance']),
                "total_sprints": team_a_stats['total_sprints']
            },
            "team_b": {
                "avg_possession_per_player": round(team_b_stats['total_possession_pct'] / max(team_b_stats['player_count'], 1), 1),
                "total_distance": int(team_b_stats['total_distance']),
                "total_sprints": team_b_stats['total_sprints']
            }
        },
        "top_performers": {
            "most_possession": [
                {
                    "player_id": pid,
                    "team": detailed_stats[pid]['team'],
                    "possession_pct": round(detailed_stats[pid]['possession_percentage'], 1),
                    "touches": detailed_stats[pid]['touches']
                }
                for pid, _ in sorted_by_possession[:3]
            ],
            "most_distance": [
                {
                    "player_id": pid,
                    "team": detailed_stats[pid]['team'],
                    "distance": int(detailed_stats[pid]['total_distance_pixels'])
                }
                for pid, _ in sorted_by_distance[:3]
            ],
            "most_sprints": [
                {
                    "player_id": pid,
                    "team": detailed_stats[pid]['team'],
                    "sprints": detailed_stats[pid]['sprint_count']
                }
                for pid, _ in sorted_by_sprints[:3]
            ]
        },
        "players_simple": {
            pid: {
                "team": stats['team'],
                "time": round(stats['time_on_screen'], 1),
                "distance": int(stats['total_distance_pixels']),
                "possession_pct": round(stats['possession_percentage'], 1),
                "touches": stats['touches'],
                "sprints": stats['sprint_count']
            }
            for pid, stats in detailed_stats.items()
        }
    }

    if progress_callback:
        progress_callback(video_info.total_frames, video_info.total_frames,
                        f"Complete! Tracked {len(detailed_stats)} players")


    return output_path, detailed_stats, llm_package