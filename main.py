from utils import read_video, write_video
from trackers import Tracker
from team_assigner import TeamAssigner


def main():
    frames = read_video("data/normal_7sec.mp4")

    tracker = Tracker("models/best8.pt")

    tracks = tracker.get_object_tracks(frames)
    tracker.add_position_to_tracks(tracks)

    tracks['ball'] = tracker.interpolate_tracks(tracks['ball'])
    # tracks['players'] = tracker.interpolate_players(tracks['players'])


    # print(tracks)
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(frames[0],tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    output_video_frames = tracker.draw_annotations(frames, tracks)

    print("Output tracks: ", tracks)

    duel_frames = tracker.detect_duel(output_video_frames, tracks)

    write_video(duel_frames, "output/test8.avi")

if __name__ == "__main__":
    main()
