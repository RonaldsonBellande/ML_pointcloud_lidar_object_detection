from header_imports import *


class classification_enviroment(gym.Env):
    def __init__(self, number_classes, image_size, data_set, image_per_episode = 1):

        self.number_classes = number_classes
        self.image_size = image_size
        self.images_per_episode = image_per_episode
        self.step_count = 0

        self.X, self.Y = data_set
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
        
        next_state_idx = random.randint(0, len(self.X) - 1)
        self.expected_action = int(self.Y[next_state_idx])
        state_space = self.X[next_state_idx]

        return state_space


    def reset(self):
        
        self.step_count = 0
        next_state = self.state()

        return next_state


        





