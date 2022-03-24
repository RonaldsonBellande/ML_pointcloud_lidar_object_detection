from header_imports import *


class classification_with_model(object):
    def __init__(self, saved_model):
        
        self.pointcloud = []
        self.number_of_points = 2048
        self.saved_model = saved_model
        self.model = keras.models.load_model("models/" + self.saved_model)
        self.path  = "PointCloud_data/"
        self.true_path = self.path + "Testing/"
        self.number_images_to_plot = 16
        self.valid_images = [".off"]
        self.graph_path = "graph_charts/" + "prediction_with_model_saved/"
        self.model_category = ['toilet', 'monitor', 'dresser', 'sofa', 'table', 'night_stand', 'chair', 'bathtub', 'bed', 'desk']
        
        self.setup_structure()
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
        self.X_test = self.pointcloud.astype("float32") / 255


    def check_valid(self, input_file):

        for img in os.listdir(self.true_path + input_file):
            ext = os.path.splitext(img)[1]
            if ext.lower() not in self.valid_images:
                continue


    def plot_prediction_with_model(self):

        plt.figure(dpi=500)
        predicted_classes = self.model.predict(self.X_test)
        
        for i in range(self.number_images_to_plot):
            plt.subplot(4,4,i+1)
            plt.axis('off')
            plt.title("Predicted - {}".format(self.model_category[np.argmax(predicted_classes[i], axis=0)]), fontsize=1)
            plt.tight_layout()
            plt.savefig(self.graph_path + "model_classification_detection_with_model_trained_prediction" + str(self.saved_model) + '.png')


    def read_file_type(self, pointcloud_data):
        
        vertice, face = self.vertices_and_faces(pointcloud_data)
        faces_area = np.zeros((len(face)))
        vertice = np.array(vertice)
        axis.plot_trisurf(vertice[:, 0], vertice[:,1], triangles=faces_area, Z=vertice[:,2])
        axis.set_title(str(pointcloud_files[34:-4]))

        return 
            
    
    def vertices_and_faces(self, pointcloud_data):
            
            n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(' ')])
            vertices = [[float(s) for s in file.readline().strip().split(' ')] for i in range(n_verts)]
            faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i in range(n_faces)]
            return vertices, faces

        
