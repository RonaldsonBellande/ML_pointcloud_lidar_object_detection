from header_import import *


class classification_enviroment(gym.Env):
    def __init__(self, image_number, image_size, image_per_episode = 1):

        self.image_number = image_number
        self.image_size = image_size
        self.images_per_episode = image_per_episode
        self.step_count = 0

        self.action_space = spaces.Discrete(self.number_classes)
        self.state_space = spaces.Box(low=0, high=1, shape=(self.image_size, self.image_size, 1), dtype=np.float32)


    def step(self, action):
        
        done = False
        reward = int(action == self.expected_action)
        next_state = self.state()

        self.step_count += 1
        if self.step_count >= self.images_per_episode:
            done = True
        
        return action, reward, next_state, done
    

    def state(self):
        
        if self.random:
            next_obs_idx = random.randint(0, len(self.x) - 1)
            self.expected_action = int(self.y[next_obs_idx])
            state_space = self.x[next_obs_idx]

        else:
            next_state = self.x[self.dataset_idx]
            self.expected_action = int(self.y[self.dataset_idx])

            self.dataset_idx += 1
            if self.dataset_idx >= len(self.x):
                raise StopIteration()

        return state_space

    def reset(self):
        
        self.step_count = 0
        next_state = self.state()

        return next_state


        





