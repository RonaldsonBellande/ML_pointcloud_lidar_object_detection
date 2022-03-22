from header_imports import *


class plot_graphs(object):
    def __init__(self):
        pass


    def plot_episode_time_step(self, data, type_graph):

        fig = plt.figure()
        axis = fig.add_subplot(111)
        color_graph = "blue"

        if type_graph == "cumulative_reward":
            axis.plot(data, color=color_graph)
            axis.set_title("Reward vs Episode")
            axis.set_xlabel("Episode")
            axis.set_ylabel("Reward per Step")
        elif type_graph == "step_number":
            axis.plot(data, color=color_graph)
            axis.set_title("Number of steps per episode vs. Episode")
            axis.set_xlabel("Episode")
            axis.set_ylabel("step per episode")
        plt.savefig(self.enviroment_path + self.algorithm_name + "_" + type_graph + ".png", dpi =500)


    def plot_model(self):

        plt.plot(self.q_learning_models.history['accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'Validation'], loc='upper left')
        plt.savefig(self.model_detail_path + self.algorithm_name + '_accuracy' + '.png', dpi =500)
        plt.clf()

        plt.plot(self.q_learning_models.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'Validation'], loc='upper left')
        plt.savefig(self.model_detail_path + self.algorithm_name + '_lost'+'.png', dpi =500)
        plt.clf()

