from header_imports import *


class classification_with_model(object):
    def __init__(self, model =  "normal_category_1_model1_computer_vision_categories_43_model.h5"):

        self.model = keras.models.load_model("models/" + model)
        self.image_path = "traffic_signs" + "/Test"
        
        self.number_classes = 43
        self.number_images_to_plot = 100

        self.prepare_image_data()
        self.plot_prediction_with_model()

        _, acc = self.model.evaluate(self.X_test, self.Y_test, verbose=1)
        print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))



    def prepare_image_data(self):

        for image in os.listdir(self.image_path):
            image_resized = cv2.imread(os.path.join(self.image_path,image))
            image_resized = cv2.resize(image_resized,(self.image_size, self.image_size), interpolation = cv2.INTER_AREA)
            self.image_file.append(image_resized)
            self.label_name.append(1)

        _, self.X_test, _, self.Y_test_vec = train_test_split(self.image_file, self.label_name, test_size = 1.0, random_state = 42)
        self.Y_test = tf.keras.utils.to_categorical(self.Y_test_vec, self.number_classes)
        self.X_test = self.X_test.astype("float32") / 255



    def plot_prediction_with_model(self):

        plt.figure(dpi=500)
        predicted_classes = self.model.predict_classes(self.X_test)

        for i in range(self.number_images_to_plot):
            plt.subplot(10,10,i+1)
            fig=plt.imshow(self.X_test[i,:,:,:])
            plt.axis('off')
            plt.title("Predicted - {}".format(self.model_categories[predicted_classes[i]]),fontsize=1)
            plt.tight_layout()
            plt.savefig("graph_charts/" + "model_classification_detection_with_model_trained/"+ "_" + '_prediction' + str(self.number_classes) + '.png')

        
