from header_imports import *

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
class pointcloud_imagery(object):
    def __init__(self):
        
        self.path  = "PointCloud_data/"
        self.true_path = self.path + "PointCloud/"
        self.valid_images = [".off"]
        self.structure()
        self.save_path = "pointcloud_visualization/" 

    
    def structure(self):

        self.category_names =  os.listdir(self.true_path)
        folder = next(os.walk(self.true_path))[1]
        self.number_classes = len(folder)

        for i in range(0, self.number_classes):
            self.check_valid(self.category_names[i])

        for i in range(0, self.number_classes):
            self.read_file_type(self.category_names[i])


    def read_file_type(self, input_file):
        figure = plt.figure()
        axis = fig.add_subplot(111, projection='3d')
        self.files = [self.true_path + input_file + '/' + i for i in os.listdir(self.true_path + '/' + input_file)]
        
        for pointcloud_files in self.files:
            vertice, face = self.vertices_and_faces(pointcloud_files)

            faces_area = np.zeros((len(face)))
            vertice = np.array(vertice)
            
            self.save_path = "pointcloud_visualization/" 
            axis.plot_trisurf(vertice[:, 0], vertice[:,1], triangles=faces_area, Z=vertice[:,2])
            figure.savefig(str(self.save_path) + str(input_file) + "/" + " 1" + '.png', dpi =500)

    def check_valid(self, input_file):
        for img in os.listdir(self.true_path + input_file):
            ext = os.path.splitext(img)[1]
            if ext.lower() not in self.valid_images:
                continue


    def vertices_and_faces(self, file_name):
        with open(file_name, 'r') as file:
            if 'OFF' != file.readline().strip():
                raise('Not a valid OFF header')
            
            n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(' ')])
            vertices = [[float(s) for s in file.readline().strip().split(' ')] for i in range(n_verts)]
            faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i in range(n_faces)]
            return vertices, faces
