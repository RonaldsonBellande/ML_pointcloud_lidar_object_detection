from header_imports import *


class classification_with_model(object):
    def __init__(self, model =  "model1_computer_vision_categories_10_model.h5"):

        self.model = keras.models.load_model("models/" + model)
        self.path  = "PointCloud_data/"
        self.true_path = self.path + "Testing/"
        self.number_classes = 10
        self.number_images_to_plot = 16
        self.setup_structure()
        self.splitting_data_normalize()
        self.plot_prediction_with_model()

        _, accuracy = self.model.evaluate(self.X_test, self.Y_test, verbose=1)
        print('Restored model, accuracy: {:5.2f}%'.format(100 * accuracy))


    def setup_structure(self):

        self.category_names =  os.listdir(self.true_path)
        folder = next(os.walk(self.true_path))[1]
        self.number_classes = len(folder)

        self.check_valid(self.category_names[0])
        
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

        _, self.X_test, _, self.Y_test_vec = train_test_split(self.pointcloud, self.label_name, test_size = 1, random_state = 42)
        self.Y_test = tf.keras.utils.to_categorical(self.Y_test_vec, self.number_classes)
        self.X_test = self.X_test.astype("float32") / 255


    def plot_prediction_with_model(self):

        plt.figure(dpi=500)
        predicted_classes = self.model.predict_classes(self.X_test)

        for i in range(self.number_images_to_plot):
            plt.subplot(4,4,i+1)
            plt.axis('off')
            plt.title("Predicted - {}".format(self.model_categories[predicted_classes[i]]),fontsize=1)
            plt.tight_layout()
            plt.savefig("graph_charts/" + "model_classification_detection_with_model_trained/"+ "_" + '_prediction' + str(self.number_classes) + '.png')

        
