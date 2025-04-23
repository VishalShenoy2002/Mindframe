from utils.config import load_config_yaml
import keras
import math
from typing import Any

CONFIG_DATA = load_config_yaml('./src/neuroforge/config/model-config.yml')

class XSmallClassificationNetwork:
    def __init__(self, name: str, input_shape: tuple, output_shape: int, model_type: str = "uniform", num_layers: int = 3):
        
        self.__name = name or self.__class__.__name__
        self.__input_shape = input_shape
        self.__output_shape = output_shape
        self.__config_data = CONFIG_DATA[f"{self.__class__.__name__}"]
        self.__model_type = model_type
        self.__num_layers = num_layers
        
        self.__model = None
        self.__build_model()

    def __build_model(self):
        self.__model = keras.models.Sequential(name=self.__name, trainable=True)
        self.__model.add(keras.layers.Input(shape=self.__input_shape))

        layer_config = self.__config_data['layer']
        layer_type = layer_config['type']
        neurons = layer_config['params']['neurons']
        activation = layer_config['params']['activation']
        layer_class = getattr(keras.layers, layer_type)

        if self.__model_type.lower() == "uniform":
            for _ in range(self.__num_layers):
                self.__model.add(layer_class(neurons, activation=activation))
        
        elif self.__model_type.lower() == "incremental":
            for idx in range(1, self.__num_layers + 1):
                self.__model.add(layer_class(neurons * idx, activation=activation))
        
        elif self.__model_type.lower() == "decremental":
            for idx in range(self.__num_layers, 0, -1):
                self.__model.add(layer_class(neurons * idx, activation=activation))
        
        else:
            raise ValueError(f"Invalid model_type '{self.__model_type}'. Choose from 'uniform', 'incremental', 'decremental'.")

        output_config = self.__config_data['output-layer']
        output_type = output_config['type']
        output_activation = output_config['params']['activation']
        output_class = getattr(keras.layers, output_type)
        
        self.__model.add(output_class(self.__output_shape, activation=output_activation))

        self.__model.compile(optimizer=self.__config_data['optimizer'], loss=self.__config_data['loss'], metrics=['accuracy'])


    def summary(self):
        return self.__model.summary()

    def train(self, x_train: Any, y_train: Any, epchos: int = 10, batch_size: int = 32, validation_data: Any = None, verbose: int = 1):
        history = self.__model.fit(x_train, y_train, epochs=epchos, batch_size=batch_size, validation_data=validation_data, verbose=verbose)
        return history

    def predict(self, x: Any, verbose: int = 0):
        return self.__model.predict(x, verbose=verbose)

    def save(self):
        self.__model.save(f'./ml-models/{self.__name.lower()}_shallow_nn.keras')

    @property
    def num_layers(self):
        return self.__num_layers

    @num_layers.setter
    def num_layers(self, value):
        if not isinstance(value, int):
            raise TypeError(f"Expected type 'int' but got '{type(value)}'")
        if value <= 0:
            raise ValueError("num_layers must be greater than 0.")
        
        self.__num_layers = value
        self.__build_model() 
        
    @property
    def total_params(self):
        count = self.__model.count_params()
        if count < 1000:
            return str(count)
        units = ['', 'K', 'M', 'B', 'T']
        magnitude = int(math.log10(count) // 3)
        scaled = count / (1000 ** magnitude)
        return f"{scaled:.1f}{units[magnitude]}"
    

class SmallClassificationNetwork:
    def __init__(self, name: str, input_shape: tuple, output_shape: int, model_type: str = "uniform", num_layers: int = 3):
        
        self.__name = name or self.__class__.__name__
        self.__input_shape = input_shape
        self.__output_shape = output_shape
        self.__config_data = CONFIG_DATA[f"{self.__class__.__name__}"]
        self.__model_type = model_type
        self.__num_layers = num_layers
        
        self.__model = None
        self.__build_model()

    def __build_model(self):
        self.__model = keras.models.Sequential(name=self.__name, trainable=True)
        self.__model.add(keras.layers.Input(shape=self.__input_shape))

        layer_config = self.__config_data['layer']
        layer_type = layer_config['type']
        neurons = layer_config['params']['neurons']
        activation = layer_config['params']['activation']
        layer_class = getattr(keras.layers, layer_type)

        if self.__model_type.lower() == "uniform":
            for _ in range(self.__num_layers):
                self.__model.add(layer_class(neurons, activation=activation))
        
        elif self.__model_type.lower() == "incremental":
            for idx in range(1, self.__num_layers + 1):
                self.__model.add(layer_class(neurons * idx, activation=activation))
        
        elif self.__model_type.lower() == "decremental":
            for idx in range(self.__num_layers, 0, -1):
                self.__model.add(layer_class(neurons * idx, activation=activation))
        
        else:
            raise ValueError(f"Invalid model_type '{self.__model_type}'. Choose from 'uniform', 'incremental', 'decremental'.")

        output_config = self.__config_data['output-layer']
        output_type = output_config['type']
        output_activation = output_config['params']['activation']
        output_class = getattr(keras.layers, output_type)
        
        self.__model.add(output_class(self.__output_shape, activation=output_activation))

        self.__model.compile(optimizer=self.__config_data['optimizer'], loss=self.__config_data['loss'], metrics=['accuracy'])


    def summary(self):
        return self.__model.summary()

    def train(self, x_train: Any, y_train: Any, epchos: int = 10, batch_size: int = 32, validation_data: Any = None, verbose: int = 1):
        history = self.__model.fit(x_train, y_train, epochs=epchos, batch_size=batch_size, validation_data=validation_data, verbose=verbose)
        return history

    def predict(self, x: Any, verbose: int = 0):
        return self.__model.predict(x, verbose=verbose)

    def save(self):
        self.__model.save(f'./ml-models/{self.__name.lower()}_shallow_nn.keras')

    @property
    def num_layers(self):
        return self.__num_layers

    @num_layers.setter
    def num_layers(self, value):
        if not isinstance(value, int):
            raise TypeError(f"Expected type 'int' but got '{type(value)}'")
        if value <= 0:
            raise ValueError("num_layers must be greater than 0.")
        
        self.__num_layers = value
        self.__build_model() 

    @property
    def total_params(self):
        count = self.__model.count_params()
        if count < 1000:
            return str(count)
        units = ['', 'K', 'M', 'B', 'T']
        magnitude = int(math.log10(count) // 3)
        scaled = count / (1000 ** magnitude)
        return f"{scaled:.1f}{units[magnitude]}"
    

class MediumClassificationNetwork:
    def __init__(self, name: str, input_shape: tuple, output_shape: int, model_type: str = "uniform", num_layers: int = 3):
        
        self.__name = name or self.__class__.__name__
        self.__input_shape = input_shape
        self.__output_shape = output_shape
        self.__config_data = CONFIG_DATA[f"{self.__class__.__name__}"]
        self.__model_type = model_type
        self.__num_layers = num_layers
        
        self.__model = None
        self.__build_model()

    def __build_model(self):
        self.__model = keras.models.Sequential(name=self.__name, trainable=True)
        self.__model.add(keras.layers.Input(shape=self.__input_shape))

        layer_config = self.__config_data['layer']
        layer_type = layer_config['type']
        neurons = layer_config['params']['neurons']
        activation = layer_config['params']['activation']
        layer_class = getattr(keras.layers, layer_type)

        if self.__model_type.lower() == "uniform":
            for _ in range(self.__num_layers):
                self.__model.add(layer_class(neurons, activation=activation))
        
        elif self.__model_type.lower() == "incremental":
            for idx in range(1, self.__num_layers + 1):
                self.__model.add(layer_class(neurons * idx, activation=activation))
        
        elif self.__model_type.lower() == "decremental":
            for idx in range(self.__num_layers, 0, -1):
                self.__model.add(layer_class(neurons * idx, activation=activation))
        
        else:
            raise ValueError(f"Invalid model_type '{self.__model_type}'. Choose from 'uniform', 'incremental', 'decremental'.")

        output_config = self.__config_data['output-layer']
        output_type = output_config['type']
        output_activation = output_config['params']['activation']
        output_class = getattr(keras.layers, output_type)
        
        self.__model.add(output_class(self.__output_shape, activation=output_activation))

        self.__model.compile(optimizer=self.__config_data['optimizer'], loss=self.__config_data['loss'], metrics=['accuracy'])


    def summary(self):
        return self.__model.summary()

    def train(self, x_train: Any, y_train: Any, epchos: int = 10, batch_size: int = 32, validation_data: Any = None, verbose: int = 1):
        history = self.__model.fit(x_train, y_train, epochs=epchos, batch_size=batch_size, validation_data=validation_data, verbose=verbose)
        return history

    def predict(self, x: Any, verbose: int = 0):
        return self.__model.predict(x, verbose=verbose)

    def save(self):
        self.__model.save(f'./ml-models/{self.__name.lower()}_shallow_nn.keras')

    @property
    def num_layers(self):
        return self.__num_layers

    @num_layers.setter
    def num_layers(self, value):
        if not isinstance(value, int):
            raise TypeError(f"Expected type 'int' but got '{type(value)}'")
        if value <= 0:
            raise ValueError("num_layers must be greater than 0.")
        
        self.__num_layers = value
        self.__build_model() 

    @property
    def total_params(self):
        count = self.__model.count_params()
        if count < 1000:
            return str(count)
        units = ['', 'K', 'M', 'B', 'T']
        magnitude = int(math.log10(count) // 3)
        scaled = count / (1000 ** magnitude)
        return f"{scaled:.1f}{units[magnitude]}"

class LargeClassificationNetwork:
    def __init__(self, name: str, input_shape: tuple, output_shape: int, model_type: str = "uniform", num_layers: int = 3):
        
        self.__name = name or self.__class__.__name__
        self.__input_shape = input_shape
        self.__output_shape = output_shape
        self.__config_data = CONFIG_DATA[f"{self.__class__.__name__}"]
        self.__model_type = model_type
        self.__num_layers = num_layers
        
        self.__model = None
        self.__build_model()

    def __build_model(self):
        self.__model = keras.models.Sequential(name=self.__name, trainable=True)
        self.__model.add(keras.layers.Input(shape=self.__input_shape))

        layer_config = self.__config_data['layer']
        layer_type = layer_config['type']
        neurons = layer_config['params']['neurons']
        activation = layer_config['params']['activation']
        layer_class = getattr(keras.layers, layer_type)

        if self.__model_type.lower() == "uniform":
            for _ in range(self.__num_layers):
                self.__model.add(layer_class(neurons, activation=activation))
        
        elif self.__model_type.lower() == "incremental":
            for idx in range(1, self.__num_layers + 1):
                self.__model.add(layer_class(neurons * idx, activation=activation))
        
        elif self.__model_type.lower() == "decremental":
            for idx in range(self.__num_layers, 0, -1):
                self.__model.add(layer_class(neurons * idx, activation=activation))
        
        else:
            raise ValueError(f"Invalid model_type '{self.__model_type}'. Choose from 'uniform', 'incremental', 'decremental'.")

        output_config = self.__config_data['output-layer']
        output_type = output_config['type']
        output_activation = output_config['params']['activation']
        output_class = getattr(keras.layers, output_type)
        
        self.__model.add(output_class(self.__output_shape, activation=output_activation))

        self.__model.compile(optimizer=self.__config_data['optimizer'], loss=self.__config_data['loss'], metrics=['accuracy'])


    def summary(self):
        return self.__model.summary()

    def train(self, x_train: Any, y_train: Any, epchos: int = 10, batch_size: int = 32, validation_data: Any = None, verbose: int = 1):
        history = self.__model.fit(x_train, y_train, epochs=epchos, batch_size=batch_size, validation_data=validation_data, verbose=verbose)
        return history

    def predict(self, x: Any, verbose: int = 0):
        return self.__model.predict(x, verbose=verbose)

    def save(self):
        self.__model.save(f'./ml-models/{self.__name.lower()}_shallow_nn.keras')

    @property
    def num_layers(self):
        return self.__num_layers

    @num_layers.setter
    def num_layers(self, value):
        if not isinstance(value, int):
            raise TypeError(f"Expected type 'int' but got '{type(value)}'")
        if value <= 0:
            raise ValueError("num_layers must be greater than 0.")
        
        self.__num_layers = value
        self.__build_model() 

    @property
    def total_params(self):
        count = self.__model.count_params()
        if count < 1000:
            return str(count)
        units = ['', 'K', 'M', 'B', 'T']
        magnitude = int(math.log10(count) // 3)
        scaled = count / (1000 ** magnitude)
        return f"{scaled:.1f}{units[magnitude]}"
    

class XLargeClassificationNetwork:
    def __init__(self, name: str, input_shape: tuple, output_shape: int, model_type: str = "uniform", num_layers: int = 3):
        
        self.__name = name or self.__class__.__name__
        self.__input_shape = input_shape
        self.__output_shape = output_shape
        self.__config_data = CONFIG_DATA[f"{self.__class__.__name__}"]
        self.__model_type = model_type
        self.__num_layers = num_layers
        
        self.__model = None
        self.__build_model()

    def __build_model(self):
        self.__model = keras.models.Sequential(name=self.__name, trainable=True)
        self.__model.add(keras.layers.Input(shape=self.__input_shape))

        layer_config = self.__config_data['layer']
        layer_type = layer_config['type']
        neurons = layer_config['params']['neurons']
        activation = layer_config['params']['activation']
        layer_class = getattr(keras.layers, layer_type)

        if self.__model_type.lower() == "uniform":
            for _ in range(self.__num_layers):
                self.__model.add(layer_class(neurons, activation=activation))
        
        elif self.__model_type.lower() == "incremental":
            for idx in range(1, self.__num_layers + 1):
                self.__model.add(layer_class(neurons * idx, activation=activation))
        
        elif self.__model_type.lower() == "decremental":
            for idx in range(self.__num_layers, 0, -1):
                self.__model.add(layer_class(neurons * idx, activation=activation))
        
        else:
            raise ValueError(f"Invalid model_type '{self.__model_type}'. Choose from 'uniform', 'incremental', 'decremental'.")

        output_config = self.__config_data['output-layer']
        output_type = output_config['type']
        output_activation = output_config['params']['activation']
        output_class = getattr(keras.layers, output_type)
        
        self.__model.add(output_class(self.__output_shape, activation=output_activation))

        self.__model.compile(optimizer=self.__config_data['optimizer'], loss=self.__config_data['loss'], metrics=['accuracy'])


    def summary(self):
        return self.__model.summary()

    def train(self, x_train: Any, y_train: Any, epchos: int = 10, batch_size: int = 32, validation_data: Any = None, verbose: int = 1):
        history = self.__model.fit(x_train, y_train, epochs=epchos, batch_size=batch_size, validation_data=validation_data, verbose=verbose)
        return history

    def predict(self, x: Any, verbose: int = 0):
        return self.__model.predict(x, verbose=verbose)

    def save(self):
        self.__model.save(f'./ml-models/{self.__name.lower()}_shallow_nn.keras')

    @property
    def num_layers(self):
        return self.__num_layers

    @num_layers.setter
    def num_layers(self, value):
        if not isinstance(value, int):
            raise TypeError(f"Expected type 'int' but got '{type(value)}'")
        if value <= 0:
            raise ValueError("num_layers must be greater than 0.")
        
        self.__num_layers = value
        self.__build_model() 

    @property
    def total_params(self):
        count = self.__model.count_params()
        if count < 1000:
            return str(count)
        units = ['', 'K', 'M', 'B', 'T']
        magnitude = int(math.log10(count) // 3)
        scaled = count / (1000 ** magnitude)
        return f"{scaled:.1f}{units[magnitude]}"


xl_nn = XLargeClassificationNetwork("XL Classification Network",(10,),10,model_type='incremental',num_layers=100)
print(xl_nn.total_params)
xl_nn.summary()