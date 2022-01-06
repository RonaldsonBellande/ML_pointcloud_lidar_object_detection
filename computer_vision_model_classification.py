from header_imports import *


class classification_with_model(object):
    def __init__(self, save_model):
        
        self.pointcloud = []
        self.number_of_points = 2048
        self.save_type = save_model
        self.model = keras.models.load_model("models/" + self.save_type)
        self.path  = "PointCloud_data/"
        self.true_path = self.path + "Testing/"
        self.number_images_to_plot = 16
        self.valid_images = [".off"]
        self.graph_path = "graph_charts/" + "prediction_with_model_saved/"
        self.model_categpory = ['toilet', 'monitor', 'dresser', 'sofa', 'table', 'night_stand', 'chair', 'bathtub', 'bed', 'desk']
        
        self.setup_structure()
        self.splitting_data_normalize()
        self.plot_prediction_with_model()


    def setup_structure(self):

        self.category_names =  os.listdir(self.true_path)
        folder = next(os.walk(self.true_path))[1]
        self.number_classes = len(folder)
        
        for i in range(self.number_classes):
            self.check_valid(self.category_names[i])
        
        for label in self.category_names:
            self.pointcloud_file = [self.true_path + label + '/' + i for i in os.listdir(self.true_path + '/' + label)]
            for point in self.pointcloud_file:
                self.pointcloud.append(trimesh.load(point).sample(self.number_of_points))
        
        self.pointcloud = np.array(self.pointcloud)
        self.pointcloud =  self.pointcloud.reshape(self.pointcloud.shape[0], self.pointcloud.shape[1], self.pointcloud.shape[2], 1)

    def check_valid(self, input_file):

        for img in os.listdir(self.true_path + input_file):
            ext = os.path.splitext(img)[1]
            if ext.lower() not in self.valid_images:
                continue


    def splitting_data_normalize(self):
        self.X_test = self.pointcloud.astype("float32") / 255


    def plot_prediction_with_model(self):

        plt.figure(dpi=500)
        predicted_classes = self.model.predict(self.X_test)
        
        for i in range(self.number_images_to_plot):
            plt.subplot(4,4,i+1)
            plt.axis('off')
            plt.title("Predicted - {}".format(self.model_categpory[np.argmax(predicted_classes[i], axis=0)]), fontsize=1)
            plt.tight_layout()
            plt.savefig(self.graph_path + "model_classification_detection_with_model_trained_prediction" + str(self.save_type) + '.png')

        
