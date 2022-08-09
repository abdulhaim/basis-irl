from typing import Tuple

from gym.envs.registration import register
import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, CircularLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.controller import MDPVehicle
import random

class RoundaboutEnv(AbstractEnv):
    DISTANCE_WANTED = 5.0 + ControlledVehicle.LENGTH  # [m]
    """Desired jam distance to the front vehicle."""
    TIME_WANTED = 1.5  # [s]
    """Desired time gap to the front vehicle."""
    COMFORT_ACC_MAX = 3.0  # [m/s2]
    """Desired maximum acceleration."""
    COMFORT_ACC_MIN = -5.0  # [m/s2]
    """Desired maximum deceleration."""
    REWARD_MIN = 0
    REWARD_MAX = 1

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "absolute": True,
                "features_range": {"x": [-100, 100], "y": [-100, 100], "vx": [-15, 15], "vy": [-15, 15]},
            },
            "action": {
                "type": "DiscreteMetaAction",
                "target_speeds": [0, 8, 16, 24, 32]
            },
            "incoming_vehicle_destination": None,
            "collision_reward": -1,
            "high_speed_reward": 0.2,
            "distance_reward": -0.3,
            "vehicles_count": 10,
            "right_lane_reward": 0.5,
            "lane_change_reward": -0.05,
            "screen_width": 600,
            "screen_height": 600,
            "centering_position": [0.5, 0.6],
            "duration": 15,
            "normalize_reward": True
        })
        return config

    def desired_gap(self, ego_vehicle: Vehicle, front_vehicle: Vehicle = None, projected: bool = True) -> float:
        """
        Compute the desired distance between a vehicle and its leading vehicle.
        :param ego_vehicle: the vehicle being controlled
        :param front_vehicle: its leading vehicle
        :param projected: project 2D velocities in 1D space
        :return: the desired distance between the two [m]
        """
        d0 = self.config["desired_distance"] + ControlledVehicle.LENGTH 
        tau = self.TIME_WANTED
        ab = - self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN
        dv = np.dot(ego_vehicle.velocity - front_vehicle.velocity, ego_vehicle.direction) if projected \
            else ego_vehicle.speed - front_vehicle.speed
        d_star = d0 + ego_vehicle.speed * tau + ego_vehicle.speed * dv / (2 * np.sqrt(ab))
        return d_star

    def _reward(self, action: int) -> float:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        if self.config["lane_preference"] == "left":
            if lane == 0:
                lane_preferred = 2
            else:
                lane_preferred = 0
        else:
            lane_preferred = lane

        lane_change = action == 0 or action == 2
 
        collision_reward = self.config["collision_reward"] * self.vehicle.crashed 

        # Speed Reward
        maximum_speed_error = [0, self.config['target_speed']]
        scaled_speed = np.clip(utils.lmap(abs(self.vehicle.speed - self.config['target_speed']), maximum_speed_error, [1, 0]), 0, 1) # scaled speed is a fraction

        # Distance From Front Car Reward
        front_vehicle, _ = self.road.neighbour_vehicles(self.vehicle)
        current_distance = self.vehicle.lane_distance_to(front_vehicle)
        if front_vehicle != None: 
            desired_gap = self.desired_gap(self.vehicle, front_vehicle)
            gap_difference = (current_distance - desired_gap)**2

            distance_error = utils.lmap(gap_difference, [0, 2000], [0, 1])
        else:
            gap_difference = 0
            distance_error = 0

        reward = \
            + self.config["collision_reward"] * self.vehicle.crashed \
            + self.config["right_lane_reward"] * lane_preferred / max(len(neighbours) - 1, 1) \
            + self.config["high_speed_reward"] * scaled_speed \
            + self.config["distance_reward"] * distance_error \
            + self.config["lane_change_reward"] * lane_change

        if reward < self.REWARD_MIN:
            self.REWARD_MIN = reward
        elif reward > self.REWARD_MAX:
            self.REWARD_MAX = reward

        reward = 0 if not self.vehicle.on_road else reward
        return [reward, self.REWARD_MIN, self.REWARD_MAX, self.config["collision_reward"] * self.vehicle.crashed,
                                                          lane_preferred / max(len(neighbours) - 1, 1),
                                                          self.config["high_speed_reward"] * scaled_speed,
                                                          gap_difference]


    def get_expert_reward(self, action):
        return self._reward(action)[0]

    def _is_terminal(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return self.vehicle.crashed or self.steps >= self.config["duration"]

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        # Circle lanes: (s)outh/(e)ast/(n)orth/(w)est (e)ntry/e(x)it.
        center = [0, 0]  # [m]
        radius = 20  # [m]
        alpha = 24  # [deg]

        net = RoadNetwork()
        radii = [radius, radius+4]
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        line = [[c, s], [n, c]]
        for lane in [0, 1]:
            net.add_lane("se", "ex",
                         CircularLane(center, radii[lane], np.deg2rad(90 - alpha), np.deg2rad(alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("ex", "ee",
                         CircularLane(center, radii[lane], np.deg2rad(alpha), np.deg2rad(-alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("ee", "nx",
                         CircularLane(center, radii[lane], np.deg2rad(-alpha), np.deg2rad(-90 + alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("nx", "ne",
                         CircularLane(center, radii[lane], np.deg2rad(-90 + alpha), np.deg2rad(-90 - alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("ne", "wx",
                         CircularLane(center, radii[lane], np.deg2rad(-90 - alpha), np.deg2rad(-180 + alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("wx", "we",
                         CircularLane(center, radii[lane], np.deg2rad(-180 + alpha), np.deg2rad(-180 - alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("we", "sx",
                         CircularLane(center, radii[lane], np.deg2rad(180 - alpha), np.deg2rad(90 + alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("sx", "se",
                         CircularLane(center, radii[lane], np.deg2rad(90 + alpha), np.deg2rad(90 - alpha),
                                      clockwise=False, line_types=line[lane]))

        # Access lanes: (r)oad/(s)ine
        access = 170  # [m]
        dev = 85  # [m]
        a = 5  # [m]
        delta_st = 0.2*dev  # [m]

        delta_en = dev-delta_st
        w = 2*np.pi/dev
        net.add_lane("ser", "ses", StraightLane([2, access], [2, dev/2], line_types=(s, c)))
        net.add_lane("ses", "se", SineLane([2+a, dev/2], [2+a, dev/2-delta_st], a, w, -np.pi/2, line_types=(c, c)))
        net.add_lane("sx", "sxs", SineLane([-2-a, -dev/2+delta_en], [-2-a, dev/2], a, w, -np.pi/2+w*delta_en, line_types=(c, c)))
        net.add_lane("sxs", "sxr", StraightLane([-2, dev / 2], [-2, access], line_types=(n, c)))

        net.add_lane("eer", "ees", StraightLane([access, -2], [dev / 2, -2], line_types=(s, c)))
        net.add_lane("ees", "ee", SineLane([dev / 2, -2-a], [dev / 2 - delta_st, -2-a], a, w, -np.pi / 2, line_types=(c, c)))
        net.add_lane("ex", "exs", SineLane([-dev / 2 + delta_en, 2+a], [dev / 2, 2+a], a, w, -np.pi / 2 + w * delta_en, line_types=(c, c)))
        net.add_lane("exs", "exr", StraightLane([dev / 2, 2], [access, 2], line_types=(n, c)))

        net.add_lane("ner", "nes", StraightLane([-2, -access], [-2, -dev / 2], line_types=(s, c)))
        net.add_lane("nes", "ne", SineLane([-2 - a, -dev / 2], [-2 - a, -dev / 2 + delta_st], a, w, -np.pi / 2, line_types=(c, c)))
        net.add_lane("nx", "nxs", SineLane([2 + a, dev / 2 - delta_en], [2 + a, -dev / 2], a, w, -np.pi / 2 + w * delta_en, line_types=(c, c)))
        net.add_lane("nxs", "nxr", StraightLane([2, -dev / 2], [2, -access], line_types=(n, c)))

        net.add_lane("wer", "wes", StraightLane([-access, 2], [-dev / 2, 2], line_types=(s, c)))
        net.add_lane("wes", "we", SineLane([-dev / 2, 2+a], [-dev / 2 + delta_st, 2+a], a, w, -np.pi / 2, line_types=(c, c)))
        net.add_lane("wx", "wxs", SineLane([dev / 2 - delta_en, -2-a], [-dev / 2, -2-a], a, w, -np.pi / 2 + w * delta_en, line_types=(c, c)))
        net.add_lane("wxs", "wxr", StraightLane([-dev / 2, -2], [-access, -2], line_types=(n, c)))

        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :return: the ego-vehicle
        """
        position_deviation = 2
        speed_deviation = 2

        # Ego-vehicle
        ego_lane = self.road.network.get_lane(("ser", "ses", 0))
        ego_vehicle = self.action_type.vehicle_class(self.road,
                                                     ego_lane.position(125, 0),
                                                     speed=8,
                                                     heading=ego_lane.heading_at(140))
        try:
            ego_vehicle.plan_route_to("nxs")
        except AttributeError:
            pass
        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        # Incoming vehicle
        destinations = ["exr", "sxr", "nxr"]
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle = other_vehicles_type.make_on_lane(self.road,
                                                   ("we", "sx", 1),
                                                   longitudinal=5 + self.np_random.randn()*position_deviation,
                                                   speed=16 + self.np_random.randn() * speed_deviation)

        if self.config["incoming_vehicle_destination"] is not None:
            destination = destinations[self.config["incoming_vehicle_destination"]]
        else:
            destination = self.np_random.choice(destinations)
        vehicle.plan_route_to(destination)
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)

        # Other vehicles
        for i in list(range(1, 2)) + list(range(-1, 0)):
            vehicle = other_vehicles_type.make_on_lane(self.road,
                                                       ("we", "sx", 0),
                                                       longitudinal=20*i + self.np_random.randn()*position_deviation,
                                                       speed=16 + self.np_random.randn() * speed_deviation)
            vehicle.plan_route_to(self.np_random.choice(destinations))
            vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)

        # Entering vehicle
        vehicle = other_vehicles_type.make_on_lane(self.road,
                                                   ("eer", "ees", 0),
                                                   longitudinal=50 + self.np_random.randn() * position_deviation,
                                                   speed=16 + self.np_random.randn() * speed_deviation)
        vehicle.plan_route_to(self.np_random.choice(destinations))
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)



def create_envs(config):
    speed_range = [20, 30]
    desired_distance = [1, 10]
    lane_preference = ["left", "right"]
    envs = []
    config_dict = dict()

    if config.mode == 'expert' or config.mode == 'play-expert':
        random.seed(13 + config.seed)

    for i in range(config.env_num_tasks):
        env = RoundaboutEnv()
        env.config["lane_preference"] = random.choice(lane_preference)
        env.config["target_speed"] = random.uniform(speed_range[0], speed_range[1])
        env.config["desired_distance"] = random.uniform(desired_distance[0], desired_distance[1])

        config_dict["pref_lane_preference_env_" + str(i)] = env.config["lane_preference"]
        config_dict["pref_target_speed_env_" + str(i)] = env.config["target_speed"]
        config_dict["pref_desired_distance_env_" + str(i)] = env.config["desired_distance"]

        print("lane_preference", env.config["lane_preference"])
        print("target_speed", env.config["target_speed"])
        print("desired_distance", env.config["desired_distance"])
        env.observation_shape = env.reset().shape
        envs.append(env)

    config.update(config_dict, allow_val_change=True)
    return envs