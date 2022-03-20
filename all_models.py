from header_import import *

class models(object):
    def create_models_1(self):

        model = Sequential()
        model.add(Conv2D(filters=64,kernel_size=(7,7), strides = (1,1), padding="same", input_shape = self.input_shape, activation = "relu"))
        model.add(Dropout(0.25))
        model.add(Conv2D(filters=32,kernel_size=(7,7), strides = (1,1), padding="same", activation = "relu"))
        model.add(Dropout(0.25))
        model.add(Conv2D(filters=16,kernel_size=(7,7), strides = (1,1), padding="same", activation = "relu"))
        model.add(MaxPooling2D(pool_size = (1,1)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(units = self.number_classes, activation = "softmax", input_dim=2))
        model.compile(loss = "binary_crossentropy", optimizer="adam", metrics=["accuracy"])

        return model

    
    def create_models_2(self):

        model = Sequential()
        model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu", input_shape = self.input_shape))
        model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same",activation="relu"))
        model.add(Dropout(rate=0.25))
        model.add(Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding="same",activation="relu"))
        model.add(MaxPooling2D(pool_size = (1,1)))
        model.add(Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), padding="same",activation="relu"))
        model.add(Dropout(rate=0.25))
        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(Dropout(rate=0.5))
        model.add(Dense(units = self.number_classes, activation="softmax"))
        model.compile(loss = 'binary_crossentropy', optimizer ='adam', metrics= ['accuracy'])
	
        return model


    def create_model_3(self):

        model = Sequential()
        self.MyConv(first = True)
        self.MyConv()
        self.MyConv()
        self.MyConv()
        model.add(Flatten())
        model.add(Dense(units = self.number_classes, activation = "softmax", input_dim=2))
        model.compile(loss = "binary_crossentropy", optimizer ="adam", metrics= ["accuracy"])
        
        return model
        

    def MyConv(self, first = False):
        if first == False:
            model.add(Conv2D(64, (4, 4),strides = (1,1), padding="same", input_shape = self.input_shape))
        else:
            model.add(Conv2D(64, (4, 4),strides = (1,1), padding="same", input_shape = self.input_shape))
    
        model.add(Activation("relu"))
        model.add(Dropout(0.5))
        model.add(Conv2D(32, (4, 4),strides = (1,1),padding="same"))
        model.add(Activation("relu"))
        model.add(Dropout(0.25))


