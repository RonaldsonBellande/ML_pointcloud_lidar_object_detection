from header_imports import *


class classification_with_model(object):
    def __init__(self, model =  "model1_computer_vision_categories_10_model.h5"):
        
        self.pointcloud = []
        self.label_name = []
        self.number_of_points = 2048
        self.model = keras.models.load_model("models/" + model)
        self.path  = "PointCloud_data/"
        self.true_path = self.path + "Testing/"
        self.number_images_to_plot = 16
        self.valid_images = [".off"]
        self.labelencoder = LabelEncoder()
        self.graph_charts = "graph_charts/" + "prediction_with_model_saved/"
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
                self.label_name.append(label)
        
        self.label_name = self.labelencoder.fit_transform(self.label_name)
        self.pointcloud = np.array(self.pointcloud)
        self.pointcloud =  self.pointcloud.reshape(self.pointcloud.shape[0], self.pointcloud.shape[1], self.pointcloud.shape[2], 1)
        self.label_name = np.array(self.label_name)
        self.label_name = tf.keras.utils.to_categorical(self.label_name , num_classes=self.number_classes)


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
            plt.savefig(self.graph_charts + "model_classification_detection_with_model_trained_prediction" + '.png')

        
