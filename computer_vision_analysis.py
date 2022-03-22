from header_imports import *

if __name__ == "__main__":
    
    if len(sys.argv) != 1:

        if sys.argv[1] == "model_building":
            computer_vision__analysis_obj = computer_vision_building(model_type=sys.argv[2])


        if sys.argv[1] == "model_training":
            computer_vision_analysis_obj = computer_vision_training(model_type=sys.argv[2])
        

        if sys.argv[1] == "pointcloud_prediction":
            if sys.argv[2] == "model1":
                input_model = "model1_computer_vision_categories_10_model.h5"
            elif sys.argv[2] == "model2":
                input_model = "model2_computer_vision_categories_10_model.h5"
            elif sys.argv[2] == "model3":
                input_model = "model3_computer_vision_categories_10_model.h5"

            computer_vision_analysis_obj = classification_with_model(save_model=input_model)


        if sys.argv[1] == "transfer_learning":
            if sys.argv[2] == "model1":
                input_model = "model1_computer_vision_categories_10_model.h5"
            elif sys.argv[2] == "model2":
                input_model = "model2_computer_vision_categories_10_model.h5"
            elif sys.argv[2] == "model3":
                input_model = "model3_computer_vision_categories_10_model.h5"
            
            computer_vision_analysis_obj = transfer_learning(save_model=input_model, model_type=sys.argv[3])


        if sys.argv[1] == "pointcloud_visual":
            computer_vision_analysis_obj = pointcloud_imagery()


        if sys.argv[1] == "continuous_learning":
            if sys.argv[2] == "model1":
                input_model = "model1_computer_vision_categories_10_model.h5"
            elif sys.argv[2] == "model2":
                input_model = "model2_computer_vision_categories_10_model.h5"
            elif sys.argv[2] == "model3":
                input_model = "model3_computer_vision_categories_10_model.h5"

            computer_vision_analysis_obj = continuous_learning(save_model=input_model, model_type=sys.argv[3], episode=10, algorithm_name="double_deep_q_learning", transfer_learning="true")
            computer_vision_analysis_obj.deep_q_learning()



        if sys.argv[1] == "segmentation":
            if sys.argv[2] == "model1":
                input_model = "model1_computer_vision_categories_10_model.h5"
            elif sys.argv[2] == "model2":
                input_model = "model2_computer_vision_categories_10_model.h5"
            elif sys.argv[2] == "model3":
                input_model = "model3_computer_vision_categories_10_model.h5"




