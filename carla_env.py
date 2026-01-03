"""
CARLA Gym Environment for Reinforcement Learning
Supports both RGB and Semantic Segmentation camera inputs
"""

import gym
import gym.spaces
import numpy as np
import cv2
import torch
from collections import deque
import carla


class CarlaGymEnv(gym.Env):
    """
    Custom Gym Environment for CARLA simulator.
    Supports RGB and Semantic Segmentation camera inputs.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, host='127.0.0.1', port=2000, image_dim=84,
                 camera_type='seg', town='Town01', render_mode=False):
        """
        Initialize the CARLA Gym environment.

        Args:
            host: CARLA server IP address
            port: CARLA server port
            image_dim: Dimension for resized image (square)
            camera_type: 'rgb' or 'seg' (semantic segmentation)
            town: CARLA town name
            render_mode: Whether to render the environment
        """
        super(CarlaGymEnv, self).__init__()

        self.host = host
        self.port = port
        self.image_dim = image_dim
        self.camera_type = camera_type
        self.town = town
        self.render_mode = render_mode

        # Connect to CARLA
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()

        # Action space: 4 discrete actions
        # 0: Straight, 1: Left, 2: Right, 3: Brake
        self.action_space = gym.spaces.Discrete(4)

        # Observation space: RGB image or semantic segmentation
        if camera_type == 'rgb':
            # RGB: 3 channels
            self.observation_space = gym.spaces.Box(
                low=0, high=255,
                shape=(image_dim, image_dim, 3),
                dtype=np.uint8
            )
        else:  # semantic segmentation
            # Segmentation: 1 channel (class indices)
            self.observation_space = gym.spaces.Box(
                low=0, high=255,
                shape=(image_dim, image_dim, 1),
                dtype=np.uint8
            )

        # CARLA actors
        self.vehicle = None
        self.camera = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None

        # Sensor data storage
        self.camera_data = None
        self.collision_data = deque(maxlen=1)
        self.lane_invasion_data = deque(maxlen=1)

        # State tracking
        self.prev_velocity = 0.0
        self.current_velocity = 0.0
        self.steps = 0

        # Waypoints for spawning
        self.spawn_points = None

        self._initialize_world()

    def _initialize_world(self):
        """Initialize world settings and get spawn points."""
        # Set synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

        # Load map if needed
        if self.world.get_map().name != self.town:
            self.world = self.client.load_world(self.town)
            self.blueprint_library = self.world.get_blueprint_library()

        # Get spawn points
        self.spawn_points = self.world.get_map().get_spawn_points()

    def _spawn_vehicle(self):
        """Spawn the vehicle at a random spawn point."""
        vehicle_bp = self.blueprint_library.find('vehicle.tesla.model3')

        # Try different spawn points until one works
        for spawn_point in self.spawn_points:
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
            if self.vehicle is not None:
                break

        if self.vehicle is None:
            raise RuntimeError("Failed to spawn vehicle at any spawn point")

        # Set vehicle to autopilot initially (we'll override this)
        self.vehicle.set_autopilot(False)

    def _attach_sensors(self):
        """Attach camera and sensors to the vehicle."""
        # Camera setup
        if self.camera_type == 'rgb':
            camera_bp = self.blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', str(self.image_dim))
            camera_bp.set_attribute('image_size_y', str(self.image_dim))
        else:  # semantic segmentation
            camera_bp = self.blueprint_library.find('sensor.camera.semantic_segmentation')
            camera_bp.set_attribute('image_size_x', str(self.image_dim))
            camera_bp.set_attribute('image_size_y', str(self.image_dim))

        # Attach camera to vehicle
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera = self.world.spawn_actor(
            camera_bp, camera_transform, attach_to=self.vehicle
        )

        # Collision sensor
        collision_bp = self.blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=self.vehicle
        )

        # Lane invasion sensor
        lane_invasion_bp = self.blueprint_library.find('sensor.other.lane_invasion')
        self.lane_invasion_sensor = self.world.spawn_actor(
            lane_invasion_bp, carla.Transform(), attach_to=self.vehicle
        )

        # Register listeners
        self.camera.listen(self._process_camera_data)
        self.collision_sensor.listen(lambda event: self.collision_data.append(1))
        self.lane_invasion_sensor.listen(lambda event: self.lane_invasion_data.append(1))

    # def _process_camera_data(self, image):
    #     """Process camera sensor data and store it."""
    #     array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    #     array = np.reshape(array, (image.height, image.width, 4))

    #     # Remove alpha channel
    #     array = array[:, :, :3]

    #     # Convert to RGB if needed (CARLA provides BGR)
    #     array = array[:, :, ::-1]

    #     if self.camera_type == 'rgb':
    #         self.camera_data = array
    #     else:  # semantic segmentation
    #         # For semantic segmentation, extract class tags from Red channel
    #         # CARLA stores the CityScapes class in the R channel
    #         self.camera_data = array[:, :, 0:1]

    def _process_camera_data(self, image):
        """Process camera sensor data and store it."""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))

        # Remove alpha channel
        array = array[:, :, :3]

        # CARLA provides BGR → convert to RGB
        array = array[:, :, ::-1]

        if self.camera_type == 'rgb':
            self.camera_data = array  # RGB for RL
            self.colored_segmentation = array  # Same for RGB mode
            # Convert RGB → BGR for OpenCV display
            render_image = array[:, :, ::-1]

        else:  # semantic segmentation
            # Extract class IDs (Cityscapes class stored in Red channel)
            class_ids = array[:, :, 0]
            self.camera_data = class_ids[..., None]  # (H, W, 1) for RL

            # Correct CARLA / Cityscapes color map (RGB)
            color_map = {
                0:  (0, 0, 0),
                1:  (70, 70, 70),
                2:  (190, 153, 153),
                3:  (72, 0, 90),
                4:  (220, 20, 60),
                5:  (153, 153, 153),
                6:  (157, 234, 50),
                7:  (128, 64, 128),
                8:  (244, 35, 232),
                9:  (107, 142, 35),
                10: (0, 0, 142),
                11: (102, 102, 156),
                12: (220, 220, 0),
                13: (70, 130, 180),
                14: (81, 0, 81),
                15: (150, 100, 100),
                16: (230, 150, 140),
                17: (180, 165, 180),
                18: (250, 170, 30),
                19: (110, 190, 160),
                20: (170, 120, 50),
                21: (45, 60, 150),
                22: (145, 170, 100),
            }

            # Create colored segmentation image (RGB)
            colored_segmentation = np.zeros(
                (image.height, image.width, 3), dtype=np.uint8
            )

            for class_id, color in color_map.items():
                colored_segmentation[class_ids == class_id] = color

            self.colored_segmentation = colored_segmentation  # Store RGB version
            # Convert RGB → BGR for OpenCV display
            render_image = colored_segmentation[:, :, ::-1]

        # Scale visualization (both RGB and segmentation)
        scale = 10
        self.render_image = cv2.resize(
            render_image,
            (render_image.shape[1] * scale, render_image.shape[0] * scale),
            interpolation=cv2.INTER_NEAREST
        )

    def _preprocess_observation(self):
        """Preprocess the camera data for the neural network."""
        if self.camera_data is None:
            return np.zeros((self.image_dim, self.image_dim, 1 if self.camera_type == 'seg' else 3), dtype=np.uint8)

        obs = self.camera_data.copy()

        if self.camera_type == 'seg':
            # Normalize semantic segmentation values
            # CityScapes classes range from 0 to ~30
            obs = obs.astype(np.float32) / 255.0
        else:
            # For RGB, normalize to [0, 1]
            obs = obs.astype(np.float32) / 255.0

        return obs

    def _get_velocity(self):
        """Get current vehicle velocity in km/h."""
        if self.vehicle is None:
            return 0.0
        velocity = self.vehicle.get_velocity()
        return (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5 * 3.6  # Convert to km/h

    def _get_lane_center_deviation(self):
        """
        Calculate deviation from lane center.
        Returns distance from the center of the current lane.
        """
        if self.vehicle is None:
            return 0.0

        vehicle_location = self.vehicle.get_location()
        map_ = self.world.get_map()
        waypoint = map_.get_waypoint(vehicle_location)

        if waypoint is None:
            return 0.0

        # Get lane center
        lane_center = waypoint.transform.location

        # Calculate distance to lane center
        deviation = vehicle_location.distance(lane_center)

        return deviation

    def _calculate_reward(self, action):
        """
        Calculate the reward based on various factors.

        Reward components:
        - Velocity reward: Higher velocity is better
        - Lane center reward: Staying in lane center is better
        - Collision penalty: Large negative reward for collisions
        - Lane invasion penalty: Negative reward for leaving lane
        - Action penalty: Small penalty for braking
        """
        reward = 0.0

        # Velocity reward (encourage forward movement)
        velocity_kmh = self._get_velocity()
        reward += velocity_kmh * 0.1  # Scale factor

        # Lane center reward (encourage staying in center)
        deviation = self._get_lane_center_deviation()
        reward -= deviation * 0.5  # Penalize deviation

        # Collision penalty
        if len(self.collision_data) > 0 and self.collision_data[-1] == 1:
            reward -= 100.0

        # Lane invasion penalty
        if len(self.lane_invasion_data) > 0 and self.lane_invasion_data[-1] == 1:
            reward -= 10.0

        # Brake penalty
        if action == 3:  # Brake action
            reward -= 1.0

        # Small positive reward for surviving
        reward += 0.1

        return reward

    def _apply_action(self, action):
        """Apply the selected action to the vehicle."""
        control = carla.VehicleControl()

        if action == 0:  # Straight
            control.throttle = 0.5
            control.steer = 0.0
            control.brake = 0.0
        elif action == 1:  # Left
            control.throttle = 0.4
            control.steer = -0.5
            control.brake = 0.0
        elif action == 2:  # Right
            control.throttle = 0.4
            control.steer = 0.5
            control.brake = 0.0
        elif action == 3:  # Brake
            control.throttle = 0.0
            control.steer = 0.0
            control.brake = 1.0

        self.vehicle.apply_control(control)

    def reset(self):
        """Reset the environment to initial state."""
        # Clean up existing actors
        self._cleanup()

        # Clear sensor data
        self.camera_data = None
        self.collision_data.clear()
        self.lane_invasion_data.clear()

        # Spawn new vehicle
        self._spawn_vehicle()
        self._attach_sensors()

        # Reset state
        self.prev_velocity = 0.0
        self.current_velocity = 0.0
        self.steps = 0

        # Tick to get initial observation
        self.world.tick()
        self.world.tick()  # Extra tick to ensure sensor data

        observation = self._preprocess_observation()

        if self.render_mode:
            self.render()

        return observation

    def step(self, action):
        """Execute one step in the environment."""
        self._apply_action(action)

        # Tick the simulation
        self.world.tick()

        # Wait for sensor data
        self.world.tick()

        # Get observation
        observation = self._preprocess_observation()

        # Calculate reward
        reward = self._calculate_reward(action)

        # Check if episode is done
        self.prev_velocity = self.current_velocity
        self.current_velocity = self._get_velocity()
        self.steps += 1

        # Episode ends on collision or after max steps
        done = False
        if len(self.collision_data) > 0 and self.collision_data[-1] == 1:
            done = True
        elif self.steps >= 1000:  # Max episode length
            done = True
        elif self.current_velocity < 0.1 and self.steps > 50:  # Stuck
            done = True

        # Additional info
        info = {
            'velocity': self.current_velocity,
            'lane_deviation': self._get_lane_center_deviation(),
            'collision': len(self.collision_data) > 0 and self.collision_data[-1] == 1,
            'lane_invasion': len(self.lane_invasion_data) > 0 and self.lane_invasion_data[-1] == 1,
        }

        if self.render_mode:
            self.render()

        return observation, reward, done, False, info

    def render(self, mode='human'):
        """Render the environment."""
        if self.render_image is not None:
            cv2.imshow('CARLA', self.render_image)
            cv2.waitKey(1)

    def _cleanup(self):
        """Clean up CARLA actors."""
        if self.camera is not None:
            self.camera.destroy()
            self.camera = None

        if self.collision_sensor is not None:
            self.collision_sensor.destroy()
            self.collision_sensor = None

        if self.lane_invasion_sensor is not None:
            self.lane_invasion_sensor.destroy()
            self.lane_invasion_sensor = None

        if self.vehicle is not None:
            self.vehicle.destroy()
            self.vehicle = None

    def close(self):
        """Close the environment and clean up resources."""
        self._cleanup()
        cv2.destroyAllWindows()

        # Reset CARLA settings
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)


def create_carla_env(**kwargs):
    """Factory function to create CARLA environment."""
    return CarlaGymEnv(**kwargs)
