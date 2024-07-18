from ultralytics import YOLO
import supervision as sv
import cv2
from utils import get_bbox_width, get_center_of_bbox, get_foot_position, get_measure_distance
import pandas as pd
import numpy as np

class Tracker():

    def __init__(self, model_path) -> None:
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detection_batch = self.model.predict(frames[i:i+batch_size], conf=0.15)
            detections += detection_batch
        # print("Detections: ", detections)
        return detections

    def add_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in  track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position = get_center_of_bbox(bbox) 
                    else:
                        position = get_foot_position(bbox)

                    tracks[object][frame_num][track_id]['position'] = position
    
    def interpolate_tracks(self, track_positions):
        # Extract 'bbox' from each dictionary
        bbox_values = [list(item.values())[0]['bbox'] if item else [None, None, None, None] for item in track_positions]

        # Create DataFrame
        df_track_positions = pd.DataFrame(bbox_values, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate missing values
        df_track_positions = df_track_positions.interpolate()
        df_track_positions = df_track_positions.bfill()

        track_positions = [{1: {"bbox": x}} for x in df_track_positions.to_numpy().tolist()]

        return track_positions
        
    def interpolate_players(self, track_positions):
        all_bboxes = {}
        max_frames = len(track_positions)

        for frame_num, player_track in enumerate(track_positions):
            for player_id, track in player_track.items():
                if player_id not in all_bboxes:
                    all_bboxes[player_id] = [None] * max_frames
                all_bboxes[player_id][frame_num] = track['bbox']

        interpolated_positions = {player_id: [] for player_id in all_bboxes.keys()}

        for player_id, bboxes in all_bboxes.items():
            bboxes = [bbox if bbox is not None else [None, None, None, None] for bbox in bboxes]

            df_track_positions = pd.DataFrame(bboxes, columns=['x1', 'y1', 'x2', 'y2'])

            df_track_positions = df_track_positions.interpolate()
            df_track_positions = df_track_positions.bfill()
            df_track_positions = df_track_positions.ffill()

            interpolated_bboxes = df_track_positions.to_numpy().tolist()
            for bbox in interpolated_bboxes:
                interpolated_positions[player_id].append({"bbox": bbox})

        for frame_num in range(max_frames):
            for player_id in interpolated_positions.keys():
                if player_id in track_positions[frame_num]:
                    track_positions[frame_num][player_id] = interpolated_positions[player_id][frame_num]
                else:
                    track_positions[frame_num][player_id] = interpolated_positions[player_id][frame_num]

        return track_positions


    def get_object_tracks(self, frames):

        detections = self.detect_frames(frames)

        tracks = {
            'players': [],
            'referees': [],
            'ball': []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            detection_supervision  = sv.Detections.from_ultralytics(detection)

            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv['player']

            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks['players'].append({})
            tracks['referees'].append({})
            tracks['ball'].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player'] and track_id != 19:
                    tracks['players'][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_names_inv['referee'] or track_id == 19:
                    tracks['referees'][frame_num][track_id] = {"bbox": bbox}
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                if cls_id == cls_names_inv['ball']:
                    tracks['ball'][frame_num][track_id] = {"bbox": bbox}
            
        return tracks
    
    def has_ball_approaching(self, ball_tracks, duel_frame, duel_bbox, max_frames=240, max_distance=50):

        duel_center = get_center_of_bbox(duel_bbox)
        
        for frame_idx in range(duel_frame, min(duel_frame + max_frames, len(ball_tracks))):
            ball_bbox = ball_tracks[frame_idx][1]['bbox']
            ball_center = get_center_of_bbox(ball_bbox)
            distance = get_measure_distance(ball_center, duel_center)
            if distance < max_distance:
                return True
        return False


    def detect_duel(self, frames, tracks):
        def do_intersect(bbox1, bbox2):
            x1_min, y1_min, x1_max, y1_max = bbox1
            x2_min, y2_min, x2_max, y2_max = bbox2

            # Check if there is an intersection
            if x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min:
                return False

            return True

        duels = {}
        team_colors = []

        for player in tracks['players'][0].values():
            team_color = player['team_color']
            if not any(np.array_equal(team_color, tc) for tc in team_colors):
                team_colors.append(team_color)

        for frame_idx in range(len(frames)):
            frame_tracks = tracks['players'][frame_idx]
            player_ids_team_a = [pid for pid, player in frame_tracks.items() if np.array_equal(player['team_color'], team_colors[0])]
            player_ids_team_b = [pid for pid, player in frame_tracks.items() if np.array_equal(player['team_color'], team_colors[1])]
            
            for i in range(len(player_ids_team_a)):
                for j in range(len(player_ids_team_b)):
                    bbox1 = frame_tracks[player_ids_team_a[i]]['bbox']
                    bbox2 = frame_tracks[player_ids_team_b[j]]['bbox']
                    if do_intersect(bbox1, bbox2):
                        duel_key = (player_ids_team_a[i], player_ids_team_b[j])
                        if duel_key not in duels:
                            duels[duel_key] = []
                        duels[duel_key].append(frame_idx)

        # Filter duels that last for at least 48 frames
        filtered_duels = {k: v for k, v in duels.items() if len(v) >= 28}
        final_duels = {}
        ball_tracks = tracks['ball'] 
        print('filtered duels:', filtered_duels)

        for duel_key, frame_indices in filtered_duels.items():
            for frame_idx in frame_indices:
                bbox1 = tracks['players'][frame_idx][duel_key[0]]['bbox']
                bbox2 = tracks['players'][frame_idx][duel_key[1]]['bbox']

                if frame_idx not in final_duels:
                    final_duels[frame_idx] = {}

                # Convert coordinates to integers
                x1, y1, x2, y2 = map(int, bbox1)
                x3, y3, x4, y4 = map(int, bbox2)

                final_bbox = [x1, y1, x4, y4]

                final_duels[frame_idx][duel_key[0]] = bbox1
                final_duels[frame_idx][duel_key[1]] = bbox2
                final_duels[frame_idx]['bbox'] = final_bbox

        print('final duels', final_duels)

        real_duels = {}
        duel_keys_final = []  
        for frame_idx, duel in final_duels.items():
            duel_bbox = duel['bbox']
            if self.has_ball_approaching(ball_tracks, frame_idx, duel_bbox):
                duel_key_final = (list(duel.keys())[0], list(duel.keys())[1])
                if duel_key_final not in duel_keys_final:
                    duel_keys_final.append(duel_key_final)

            duel_key = (list(duel.keys())[0], list(duel.keys())[1])
            if duel_key in duel_keys_final:
                real_duels[frame_idx] = duel
        
        print('real duels:', real_duels)
        
        filtered_real_duels = {}
        combo_players = []

        for frame_idx, duel in real_duels.items():
            combo_player = (list(duel.keys())[0], list(duel.keys())[1])
            combo_players.append(combo_player)

        # Count occurrences of each player combo
        combo_player_counts = {combo: combo_players.count(combo) for combo in set(combo_players)}

        for frame_idx, duel in real_duels.items():
            combo_player = (list(duel.keys())[0], list(duel.keys())[1])
            if combo_player_counts[combo_player] >= 10:
                if frame_idx not in filtered_real_duels:
                    filtered_real_duels[frame_idx] = {}
                filtered_real_duels[frame_idx][combo_player[0]] = duel[combo_player[0]]
                filtered_real_duels[frame_idx][combo_player[1]] = duel[combo_player[1]]
                filtered_real_duels[frame_idx]['bbox'] = duel['bbox']

        for frame_idx, duel in filtered_real_duels.items():
            bbox = duel['bbox']

            x1, y1, x2, y2 = bbox

            cv2.rectangle(frames[frame_idx], (x1, y1), (x2, y2), (255, 0, 0), 2)

        print(f'There were {len(filtered_real_duels)} real duel frames which means {len(filtered_real_duels) / 24} seconds of real duels')

        for frame_idx, duel in filtered_real_duels.items():
            bbox = duel['bbox']
            print(f"Frame {frame_idx}: Real duel between player {list(duel.keys())[0]} and player {list(duel.keys())[1]} on coordinates {bbox}")

        return frames


    def draw_ellipse(self, frame, bbox, color, track_id = None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        cv2.ellipse(frame, center = (x_center, y2), axes = (int(width), int(0.35*width)), angle = 0.0, startAngle = -45, endAngle = 235, color = color, thickness=2,lineType=cv2.LINE_4)

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = y2 - rectangle_height // 2 + 15
        y2_rect = y2 + rectangle_height // 2 + 15

        if track_id is not None:
            cv2.rectangle(frame, (x1_rect, y1_rect), (x2_rect, y2_rect), color, cv2.FILLED)
            x1_text = x1_rect + 5
            if track_id > 99:
                x1_text -= 10
            
            cv2.putText(frame, str(track_id), (x1_text, y1_rect + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return frame

    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks['players'][frame_num]
            referee_dict = tracks['referees'][frame_num]
            ball_dict = tracks['ball'][frame_num]

            for track_id, player in player_dict.items():
                color = player.get('team_color', (0, 0, 255))
                frame = self.draw_ellipse(frame, player['bbox'], color, track_id)

                # if player.get('has_ball', False):
                #     frame = self.draw_triangle(frame, player['bbox'], (255, 0, 0))

            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee['bbox'], (0, 0, 255))
           
            for track_id, ball in ball_dict.items():
                frame = self.draw_ellipse(frame, ball['bbox'], (255, 0, 0))

            output_video_frames.append(frame)

        return output_video_frames