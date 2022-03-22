from header_imports import *


class classification_enviroment(gym.Env):
    def __init__(self, number_classes, image_size, data_set, image_per_episode):

        self.number_classes = number_classes
        self.image_size = image_size
        self.images_per_episode = image_per_episode
        self.step_count = 0

        self.X, self.Y = data_set[0], data_set[1]

        # self.Y = [j for sub in self.Y for j in sub]
        self.action_space = spaces.Discrete(self.number_classes)
        self.state_space = spaces.Box(low=0, high=1, shape=(self.X.shape[1], self.X.shape[2], 1), dtype=np.float32)


    def step(self, action):
        
        done = False
        print(action[1])
        print(self.expected_action)
        reward = int(action == self.expected_action)
        next_state = self.state()

        self.step_count += 1
        if self.step_count >= self.images_per_episode:
            done = True
        
        return action, reward, next_state, done
    

    def state(self):
        
        next_state_idx = random.randint(0, self.X.shape[2] - 1)
        # print(next_state_idx)
        self.expected_action = int(self.Y[0][next_state_idx])
        state_space = self.X[next_state_idx]

        # print(next_state_idx)
        # print(self.expected_action)
        # print(state_space)

        return state_space


    def reset(self):
        
        self.step_count = 0
        next_state = self.state()

        return next_state


