from utils.spatial_func import *
from utils.candidate_point import CandidatePoint
import pickle
import os

from utils.trajectory_func import Trajectory, STPoint
import tqdm

from datetime import *


def create_datetime(timestamp):
    timestamp = str(timestamp)
    return datetime.fromtimestamp(int(timestamp))


class ParseTraj:
    """
    ParseTraj is an abstract class for parsing trajectory.
    It defines parse() function for parsing trajectory.
    """

    def __init__(self):
        pass

    def parse(self, input_path, is_target):
        """
        The parse() function is to load data to a list of Trajectory()
        """
        pass


class ParseMMTraj(ParseTraj):
    """
    Parse map matched GPS points to trajectories list. No extra data preprocessing
    """

    def __init__(self, rn):
        super().__init__()
        self.rn = rn

    def parse(self, input_path, is_target=True):
        """
        Args:
        -----
        input_path:
            str. input directory with file name
        Returns:
        --------
        trajs:
            list. list of trajectories. trajs contain input_path file's all gps points
        """
        pickle_path = input_path.replace('.txt', '.pkl')

        pickle_root = os.path.dirname(pickle_path)
        if not os.path.exists(pickle_root):
            os.makedirs(pickle_root)

        if os.path.exists(pickle_path):
            trajs = pickle.load(open(pickle_path, 'rb'))
            return trajs

        with open(input_path, 'r') as f:
            trajs = []
            pt_list = []
            for line in tqdm.tqdm(f.readlines()):
                attrs = line.rstrip().split(' ')
                if attrs[0][0] == '-':
                    if len(pt_list) > 1:
                        traj = Trajectory(pt_list)
                        trajs.append(traj)
                    pt_list = []
                else:
                    lat = float(attrs[1])
                    lng = float(attrs[2])
                    rid = int(attrs[3])
                    if is_target:
                        projection, rate, dist = project_pt_to_road(self.rn, SPoint(lat, lng), rid)
                        candi_pt = CandidatePoint(projection.lat, projection.lng, rid, dist,
                                                  rate * self.rn.edgeDis[rid], rate)
                        pt = STPoint(lat, lng, create_datetime(str(attrs[0])), {'candi_pt': candi_pt})
                    else:
                        pt = STPoint(lat, lng, create_datetime(str(attrs[0])), {'candi_pt': None})
                    # pt contains all the attributes of class STPoint
                    pt_list.append(pt)
            if len(pt_list) > 1:
                traj = Trajectory(pt_list)
                trajs.append(traj)

        pickle.dump(trajs, open(pickle_path, 'wb+'))
        return trajs
