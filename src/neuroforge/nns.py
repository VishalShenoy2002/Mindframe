from utils.config import load_config_yaml
import keras
import math

CONFIG_DATA = load_config_yaml('./src/neuroforge/config/model-config.yml')

class XSmallClassificationNetwork:
    def __init__(self,name:str, input_shape:tuple, output_shape:int, model_type:str = "uniform", num_layers:int = 3):
        self.__name = name or self.__class__.__name__
        self.__num_layers = num_layers
        self.__input_shape = input_shape
        self.__output_shape = output_shape
        self.__config_data = CONFIG_DATA[f"{self.__class__.__name__}"]
        self.__model_type = model_type
        
        self.__model = keras.models.Sequential(name=self.__name, trainable=True)
        self.__model.add(keras.layers.Input(shape=self.__input_shape))
        
        layer_config = self.__config_data['layer']
        layer_type = layer_config['type']
        neurons = layer_config['params']['neurons']
        activation = layer_config['params']['activation']
        
        layer_class = getattr(keras.layers, layer_type)

        if self.__model_type.lower() == "uniform":
            for _ in range(self.__num_layers):
                self.__model.add(layer_class(neurons,activation=activation))
        elif self.__model_type.lower() == "incremental":
            for idx in range(1, self.__num_layers + 1):
                self.__model.add(layer_class(neurons * idx,activation=activation))
        
        elif self.__model_type.lower() == "decremental":
            for idx in range(self.__num_layers, 1, -1):
                self.__model.add(layer_class(neurons * idx,activation=activation))
        
        else:
            raise ValueError("Wrong model_type '{}'. Expected 'uniform', 'incremental' or 'decremental'.")
        
        output_config = self.__config_data['output-layer']
        output_type = output_config['type']
        activation = output_config['params']['activation']
        output_class = getattr(keras.layers, output_type)
        self.__model.add(output_class(self.__output_shape,activation=activation))
        
        self.__model.compile(optimizer=self.__config_data['optimizer'],loss=self.__config_data['loss'], metrics=['accuracy'])
        
    def summary(self):
        return self.__model.summary()
    
    @property
    def total_params(self):
        count = self.__model.count_params()
        if count < 1000:
            return str(count)
        
        units = ['', 'K', 'M', 'B', 'T']
        magnitude = int(math.log10(count) // 3)
        scaled = count / (1000 ** magnitude)
        return f"{scaled:.1f}{units[magnitude]}"
        
