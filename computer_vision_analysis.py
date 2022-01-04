from header_imports import *

if __name__ == "__main__":
    
    if len(sys.argv) != 1:

        if sys.argv[1] == "model_building":
            computer_vision__analysis_obj = computer_vision_building(model_type = sys.argv[2])

        if sys.argv[1] == "model_training":
            computer_vision_analysis_obj = computer_vision_training(model_type = sys.argv[2])
        
        if sys.argv[1] == "pointcloud_prediction":
            computer_vision_analysis_obj = classification_with_model(model = sys.argv[2])

        if sys.argv[1] == "transfer_learning":
            computer_vision_analysis_obj = computer_vision_transfer_learning(currently_build_model = sys.argv[2])

        if sys.argv[1] == "pointcloud_visual":
            computer_vision_analysis_obj = pointcloud_imagery()



