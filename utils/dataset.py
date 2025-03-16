import torch
import numpy as np
import pandas as pd
import torch.utils.data as data
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from scipy import stats
from pathlib import Path

class trajectory_dataset(data.Dataset):
    def __init__(self, root, label_path, flip=False, shift=False, sample_length=38):
        '''
        Please put each trajectory txt file in the same folder hierarchy as the clip_frames folder.
        For example, if the path of the clip_frames folder is /path/to/clip_frames/clip0, then
        the path of the trajectory txt file should be /path/to/trajectory/clip0.txt.
        ball_trajectory: FrameNorm
        ball_trajectory3: TableNorm
        '''
        self.root = Path(root)
        self.label_file = pd.read_csv(label_path, sep=' ', header=None)
        self.data_path = self.label_file[0]
        self.data_path = self.data_path.str.replace("GT_cropped_action_frames", "ball_trajectory") + ".txt"
        self.flip = flip
        self.shift = shift
        self.sample_length = sample_length

    def reomve_outlier(self, trajectory):
        x = trajectory[0]
        y = trajectory[1]
        # player = trajectory[2].iloc[0]#[0] * [trajectory.shape[0]]
        trajectory = np.array([x, y]).T

        # Remove outliers
        z = np.abs(stats.zscore(trajectory))
        outlier = (z > 1.6).any(axis=1) 
        trajectory[outlier] = np.nan
        # trajectory = trajectory[~outlier]

        out = pd.DataFrame(trajectory, columns=[0, 1])
        # out[2] = player
        out = out.interpolate(method='linear').ffill().bfill()

        return out
    
    def __get(self, idx):
        '''
        Encode:
        -> Left: -0.5, Right: 0.5
        '''
        # transfer = {'L': -0.5, 'R': 0.5}
        trajectory = pd.read_csv(self.root / self.data_path.iloc[idx], sep=' ', header=None)
        # 只取第二個落點到最後一個落點，如果是發球就全部取
        # index = trajectory[trajectory[3] == 1].index
        player = trajectory[2].iloc[0]
        # if len(index) == 3:
        #     trajectory = trajectory.iloc[index[1]:index[2]+1]
        label = torch.tensor(self.label_file.loc[idx, 2])

        x = trajectory[0].values /2 + 0.25
        y = trajectory[1].values /2 + 0.25 * 0.5625
        trajectory = self.reomve_outlier(trajectory)
        trajectory = torch.tensor(list(zip(x, y)), dtype=torch.float32)

        # # 先以左邊為基準，如果是右邊的話就將x座標反轉
        # if player == 'R':
        #     trajectory = trajectory.T
        #     trajectory[0] = 1-trajectory[0]
        #     trajectory = trajectory.T

        return trajectory, label # trajectory: (n, 3), label: (1,)
    
    def __sample(self, trajectory, length):
        '''
        The mean length of stroke actions is 34 frames. Therfore we adopt 34 frames as the length of the clip.
        (quantile(0.75) = 38) (quantile(0.8) = 41) (quantile(0.85) = 44)
        If the length of the clip is less than 34 frames, we pad zero to the end of the clip.
        If the length of the clip is more than 34 frames, we uniformly sample 34 frames from the clip.

        All the data need to first divided by 2 and then plus 0.25
        因為有先將圖片crop到桌子附近的區域，所以需要將y座標除以2，x座標加上0.25才是正確的座標
        '''
        if length < self.sample_length:
            tmp = trajectory[-1].repeat(self.sample_length-length, 1)
            trajectory = torch.cat((trajectory, tmp), axis=0)
        elif length > self.sample_length:
            index = torch.linspace(0, length-1, self.sample_length, dtype=torch.int32)
            trajectory = trajectory[index]

        if self.flip:
            trajectory = trajectory.T
            trajectory[0] = 1-trajectory[0]
            trajectory = trajectory.T
            # trajectory[2] = -trajectory[2]
        
        if self.shift:
            rand = torch.rand(1).item()/2
            trajectory = trajectory.T
            if torch.rand(1) > 0.75:
                trajectory[0] += rand
            elif torch.rand(1) > 0.5:
                trajectory[0] -= rand
            elif torch.rand(1) > 0.25:
                trajectory[1] += rand
            else:
                trajectory[1] -= rand
            trajectory = trajectory.T
        
        return trajectory
        
    def __len__(self,):
        return self.label_file.shape[0]

    def __getitem__(self, idx):
        trajectory, label = self.__get(idx)
        trajectory = self.__sample(trajectory, trajectory.shape[0])

        # add velocity x
        velocity = torch.zeros_like(trajectory)
        velocity[1:] = trajectory[1:] - trajectory[:-1]

        # add acceleration
        acceleration = torch.zeros_like(trajectory)
        acceleration[2:] = velocity[2:] - velocity[1:-1]

        trajectory = torch.cat((trajectory, velocity, acceleration), axis=1)

        return trajectory, label

if __name__ == '__main__':
    root = "../MISTTdataset/TTcompare"
    label = "../MISTTdataset/TTcompare/labels/GT_cropped_action_frames_train.txt"

    dataset = trajectory_dataset(root, label)
    for i in range(10):
        trajectory, label = dataset[i]
        print(trajectory.shape)