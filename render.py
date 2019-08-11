import numpy as np
import pandas as pd
from gym_cap.envs.agent import *
from gym_cap.envs.const import *

import os
import time
import gym
import gym_cap
from gym.wrappers import Monitor
from gym.wrappers.monitoring.video_recorder import VideoRecorder

# the modules that you can use to generate the policy.
import policy

import moviepy.editor as mp
from moviepy.video.fx.all import speedx

max_episode_length = 150

# Environment
env = gym.make("cap-v0").unwrapped  # initialize the environment
policy_red = policy.AStar()
policy_blue = policy.Fighter()

env.RENDER_INDIV_MEMORY = True
env.INDIV_MEMORY = "fog"
env.RENDER_TEAM_MEMORY = True
env.TEAM_MEMORY = "fog"
STOCH_ATTACK = True
STOCH_ATTACK_BIAS = 3

observation = env.reset(map_size=20, policy_blue=policy_blue, policy_red=policy_red)

data_dir = 'render_2'
total_run = 5
num_success = 10
num_failure = 10
vid_success = []
vid_failure = []
success_episode_num = []
failure_episode_num = []


def play_episode(frame_count, episode = 0):
    """
    play episode and render it into .gif
    """
    
    # Set video recorder
    video_dir = os.path.join(data_dir, 'raw_videos')
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if not os.path.exists(video_dir):
        os.mkdir(video_dir)
    video_fn = 'episode_' + str(episode) + '.mp4'
    video_path = os.path.join(video_dir, video_fn)

    video_recorder = VideoRecorder(env, video_path)

    # Reset environmnet
    observation = env.reset()

    # Rollout episode
    episode_length = 0.
    done = 0
    while (done == 0):
        # set exploration rate for this frame
        video_recorder.capture_frame()
        episode_length += 1

        observation, reward, done, _ = env.step()

        # stop the episode if it goes too long
        if episode_length >= max_episode_length:
            reward = -100.
            done = True

    # Closer
    video_recorder.close()
    vid = mp.VideoFileClip(video_path)

    success_flag = env.blue_win
    survival_rate = sum([agent.isAlive for agent in env.get_team_blue]) / len(env.get_team_blue)
    kill_rate = sum([not agent.isAlive for agent in env.get_team_red]) / len(env.get_team_red)

    if success_flag == 1 and len(vid_success) < num_success:
        vid_success.append(vid)
        success_episode_num.append(episode)
        
    elif success_flag == 0 and len(vid_failure) < num_failure:
        vid_failure.append(vid)
        failure_episode_num.append(episode)
    
    # rendering vid to .gif
    video_dir = os.path.join(data_dir, 'gif_videos')
    if not os.path.exists(video_dir):
        os.mkdir(video_dir)
    video_fn = 'episode_' + str(episode) + '.gif'
    video_path = os.path.join(video_dir, video_fn)
    vid.write_gif(video_path, fps=500)
    
    return episode_length, reward, frame_count + episode_length, survival_rate, kill_rate, success_flag

def render_clip(frames, filename):
    """
    The successful and failure episodes will be rendered into gif  
    """
        
    #vid = mp.concatenate_videoclips(frames)
    vid = speedx(frames, 0.1)
    final_vid = vid  # mp.clips_array([[legend, vid]])

    video_dir = os.path.join(data_dir, 'outcome_videos')
    if not os.path.exists(video_dir):
        os.mkdir(video_dir)
    video_path = os.path.join(video_dir, filename)          
    final_vid.write_gif(video_path)
        
def export_data(episode, filename):
    """
    agent's location will be exported into csv file.
    If agent is dead, leave it static
    """
    
    blue_loc = []
    red_loc = []
    file_path = os.path.join(data_dir, filename)
    for blue_agent in range(NUM_BLUE):
        if env._team_blue[blue_agent].isAlive:
            blue_loc.append(env._team_blue[blue_agent].get_loc())
        else:
            pass
    
    for red_agent in range(NUM_RED):
        if env._team_red[red_agent].isAlive:
            red_loc.append(env._team_red[red_agent].get_loc())
        else:
            pass
    
    pd.DataFrame([""]).to_csv(file_path, mode = "a", header = ["Episode {}: \n" .format(episode)])
    pd.DataFrame(blue_loc).to_csv(file_path, mode = "a", header = ["blue_loc_x", "blue_loc_y"]) 
    pd.DataFrame(red_loc).to_csv(file_path, mode = "a", header = ["red_loc_x", "red_loc_y"]) 
        
if __name__ == "__main__":
    
    start_time = time.time()
    done_flag = 0
    episode = 1
    frame_count = 0
    
    print("Episodes Progress Bars: ")
    while ((len(vid_success) < num_success) or (len(vid_failure) < num_failure)) and episode <= total_run:
        length, reward, frame_count, survival_rate, kill_rate, win = play_episode(frame_count, episode)   
        #export_data(episode, r'data.csv')        
        episode += 1
        
    # closing Ctf environment
    env.close()
    del gym.envs.registry.env_specs['cap-v0']

    print("\n Outcomes Progress Bars: ")
    if vid_success != []:
        for num, videos in enumerate(vid_success):
            video_fn_2 = 'success: episode_' + str(success_episode_num[num]) + '.gif'
            print(video_fn_2)
    else:
        print("\n No Success Episodes \n")
        
    if vid_failure != []:
        for num, videos in enumerate(vid_failure):
            video_fn_2 = 'failure: episode_' + str(failure_episode_num[num]) + '.gif'
            print(video_fn_2)
    else:
        print("\n No Failure Episodes \n")