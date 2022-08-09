import numpy as np
from gym.envs.registration import register
import gym
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
import numpy as np
import random

class HighwayEnv(AbstractEnv):

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

    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "simulation_frequency": 5,
            "lanes_count": 3,
            "vehicles_count": 20,
            "duration": 30,  # [s]
            "ego_spacing": 1.5,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "vehicles_density": 1.0,
            "collision_reward": -1.0,    # The reward received when colliding with a vehicle.
            "right_lane_reward": 1.0,  # The reward received when driving on the right-most lanes, linearly mapped to
                                       # zero for other lanes.
            "high_speed_reward": 1.0,  # The reward received when driving at full speed, linearly mapped to zero for
                                       # lower speeds according to config["speed_range"].
            "distance_reward": -0.3,
            "speed_range": [20, 30],
            "offroad_terminal": False,
            'lane_preference': 'left',
            'target_speed': 25,
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])


    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

        # Disable collision check for uncontrolled vehicles
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False

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

    def get_expert_reward(self, action: Action) -> float:
        return self._reward(action)[0]

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        # Lane Preferences Reward
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

        # Speed Reward
        maximum_speed_error = [0, self.config['target_speed']]
        vehicle_speed = self.vehicle.speed*np.cos(self.vehicle.heading) 
        scaled_speed = utils.lmap(abs(vehicle_speed - self.config['target_speed']), maximum_speed_error, [1, 0])  # scaled speed is a fraction

        # Distance From Front Car Reward
        front_vehicle, _ = self.road.neighbour_vehicles(self.vehicle)
        current_distance = self.vehicle.lane_distance_to(front_vehicle)
        desired_gap = self.desired_gap(self.vehicle, front_vehicle)
        gap_difference = (current_distance - desired_gap)**2

        distance_error = utils.lmap(gap_difference, [0, 2000], [0, 1])

        reward = \
            + self.config["collision_reward"] * self.vehicle.crashed \
            + self.config["right_lane_reward"] * lane_preferred / max(len(neighbours) - 1, 1) \
            + self.config["high_speed_reward"] * scaled_speed \
            + self.config["distance_reward"] * distance_error

        if reward < self.REWARD_MIN:
            self.REWARD_MIN = reward
        elif reward > self.REWARD_MAX:
            self.REWARD_MAX = reward

        reward = 0 if not self.vehicle.on_road else reward
        return [reward, self.REWARD_MIN, self.REWARD_MAX, self.config["collision_reward"] * self.vehicle.crashed,
                                                          lane_preferred / max(len(neighbours) - 1, 1),
                                                          self.config["high_speed_reward"] * scaled_speed,
                                                          gap_difference]

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return self.vehicle.crashed or \
            self.steps >= self.config["duration"] or \
            (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        return float(self.vehicle.crashed)


def create_envs(config):
    speed_range = [20, 30]
    desired_distance = [1, 10]
    lane_preference = ["left", "right"]
    envs = []
    config_dict = dict()

    if config.mode == 'expert' or config.mode == 'play-expert' or config.mode == "irl":
        random.seed(13 + config.seed)

    for i in range(config.env_num_tasks):
        env = HighwayEnv()
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
