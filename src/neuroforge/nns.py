from neuroforge.utils.config import load_config_yaml
import keras
import math
from typing import Any

CONFIG_DATA = load_config_yaml('model-config.yml')

class XSmallClassificationNetwork:
    """Builds and manages an extra small neural network for classification tasks."""

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
    """Builds and manages a small neural network for classification tasks."""

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
    """Builds and manages a medium-sized neural network for classification tasks."""

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
    """Builds and manages a large neural network for classification tasks."""

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
    """Builds and manages an extra large neural network for classification tasks."""

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


class XSmallImageClassificationNetwork:
    """Builds and manages an extra small convolutional neural network for image classification tasks."""

    def __init__(self, name: str, input_shape: tuple, output_shape: int, model_type: str = "uniform", num_layers: int = 3):
        self.__name = name or self.__class__.__name__
        self.__input_shape = input_shape
        self.__output_shape = output_shape
        self.__config_data = CONFIG_DATA[f"{self.__class__.__name__}"]
        self.__model_type = model_type
        self.__num_layers = num_layers

        # print(self.__config_data)

        self.__model = None
        self.__build_model()

    def __build_model(self):
        self.__model = keras.models.Sequential(name=self.__name, trainable=True)
        self.__model.add(keras.layers.Input(shape=self.__input_shape))

        layer_info = self.__config_data['layers']
        conv_layer = layer_info['conv-layer']
        pooling_layer = layer_info['pooling-layer']
        
        conv_layer_class = getattr(keras.layers,conv_layer['type'])
        conv_layer_params = conv_layer['params']
        conv_filters = conv_layer_params['filters']
        conv_activation =conv_layer_params['activation']
        conv_kernel_size = conv_layer_params['kernel-size']

        pooling_layer_class = getattr(keras.layers, pooling_layer['type'])
        pooling_layer_params = pooling_layer['params']
        pooling_size = pooling_layer_params['pool-size']

        if self.__model_type.lower() == "uniform":
            for _ in range(self.__num_layers):
                self.__model.add(conv_layer_class(conv_filters, activation=conv_activation,kernel_size=conv_kernel_size))
                self.__model.add(pooling_layer_class(pool_size=pooling_size))
        
        elif self.__model_type.lower() == "incremental":
            for idx in range(1, self.__num_layers + 1):
                self.__model.add(conv_layer_class(conv_filters * idx, activation=conv_activation,kernel_size=conv_kernel_size))
                self.__model.add(pooling_layer_class(pool_size=pooling_size))
        
        elif self.__model_type.lower() == "decremental":
            for idx in range(self.__num_layers, 0, -1):
                self.__model.add(conv_layer_class(conv_filters * idx, activation=conv_activation,kernel_size=conv_kernel_size))
                self.__model.add(pooling_layer_class(pool_size=pooling_size))
        
        else:
            raise ValueError(f"Invalid model_type '{self.__model_type}'. Choose from 'uniform', 'incremental', 'decremental'.")
        

        output_config = self.__config_data['output-layer']
        flatten_layer = output_config['flatten-layer']
        flatten_layer_class = getattr(keras.layers,flatten_layer['type']) 

        dense_layer = output_config['dense-layer']
        dense_layer_class = getattr(keras.layers,dense_layer['type'])
        dense_params = dense_layer['params']
        dense_activation = dense_params['activation']

        self.__model.add(flatten_layer_class())
        self.__model.add(dense_layer_class(self.__output_shape, activation=dense_activation))

        self.__model.compile(optimizer=self.__config_data['optimizer'], loss=self.__config_data['loss'], metrics=['accuracy'])

    def summary(self):
        return self.__model.summary()
    
    def train(self, x_train: Any, y_train: Any, epchos: int = 10, verbose:int = 1):
        history = self.__model.fit(x_train,y_train, epochs=epchos, verbose=verbose)
        return history
    
    def predict(self, x_input: Any, verbose: int = 0):
        predictions = self.__model.predict(x_input, verbose=verbose)
        return predictions

    def evaluate(self, x_test: Any, y_test: Any, verbose: int = 1):
        evaluation = self.__model.evaluate(x_test, y_test, verbose=verbose)
        return evaluation

    def save(self):
        self.__model.save(f'./ml-models/{self.__name.lower()}_xs_image_classification_network.keras')

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

class SmallImageClassificationNetwork:
    """Builds and manages a small convolutional neural network for image classification tasks."""

    def __init__(self, name: str, input_shape: tuple, output_shape: int, model_type: str = "uniform", num_layers: int = 3):
        self.__name = name or self.__class__.__name__
        self.__input_shape = input_shape
        self.__output_shape = output_shape
        self.__config_data = CONFIG_DATA[f"{self.__class__.__name__}"]
        self.__model_type = model_type
        self.__num_layers = num_layers

        # print(self.__config_data)

        self.__model = None
        self.__build_model()

    def __build_model(self):
        self.__model = keras.models.Sequential(name=self.__name, trainable=True)
        self.__model.add(keras.layers.Input(shape=self.__input_shape))

        layer_info = self.__config_data['layers']
        conv_layer = layer_info['conv-layer']
        pooling_layer = layer_info['pooling-layer']
        
        conv_layer_class = getattr(keras.layers,conv_layer['type'])
        conv_layer_params = conv_layer['params']
        conv_filters = conv_layer_params['filters']
        conv_activation =conv_layer_params['activation']
        conv_kernel_size = conv_layer_params['kernel-size']

        pooling_layer_class = getattr(keras.layers, pooling_layer['type'])
        pooling_layer_params = pooling_layer['params']
        pooling_size = pooling_layer_params['pool-size']

        if self.__model_type.lower() == "uniform":
            for _ in range(self.__num_layers):
                self.__model.add(conv_layer_class(conv_filters, activation=conv_activation,kernel_size=conv_kernel_size))
                self.__model.add(pooling_layer_class(pool_size=pooling_size))
        
        elif self.__model_type.lower() == "incremental":
            for idx in range(1, self.__num_layers + 1):
                self.__model.add(conv_layer_class(conv_filters * idx, activation=conv_activation,kernel_size=conv_kernel_size))
                self.__model.add(pooling_layer_class(pool_size=pooling_size))
        
        elif self.__model_type.lower() == "decremental":
            for idx in range(self.__num_layers, 0, -1):
                self.__model.add(conv_layer_class(conv_filters * idx, activation=conv_activation,kernel_size=conv_kernel_size))
                self.__model.add(pooling_layer_class(pool_size=pooling_size))
        
        else:
            raise ValueError(f"Invalid model_type '{self.__model_type}'. Choose from 'uniform', 'incremental', 'decremental'.")
        

        output_config = self.__config_data['output-layer']
        flatten_layer = output_config['flatten-layer']
        flatten_layer_class = getattr(keras.layers,flatten_layer['type']) 

        dense_layer = output_config['dense-layer']
        dense_layer_class = getattr(keras.layers,dense_layer['type'])
        dense_params = dense_layer['params']
        dense_activation = dense_params['activation']

        self.__model.add(flatten_layer_class())
        self.__model.add(dense_layer_class(self.__output_shape, activation=dense_activation))

        self.__model.compile(optimizer=self.__config_data['optimizer'], loss=self.__config_data['loss'], metrics=['accuracy'])

    def summary(self):
        return self.__model.summary()
    
    def train(self, x_train: Any, y_train: Any, epchos: int = 10, verbose:int = 1):
        history = self.__model.fit(x_train,y_train, epochs=epchos, verbose=verbose)
        return history
    
    def predict(self, x_input: Any, verbose: int = 0):
        predictions = self.__model.predict(x_input, verbose=verbose)
        return predictions

    def evaluate(self, x_test: Any, y_test: Any, verbose: int = 1):
        evaluation = self.__model.evaluate(x_test, y_test, verbose=verbose)
        return evaluation

    def save(self):
        self.__model.save(f'./ml-models/{self.__name.lower()}_xs_image_classification_network.keras')

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

class MediumImageClassificationNetwork:
    """Builds and manages a medium convolutional neural network for image classification tasks."""

    def __init__(self, name: str, input_shape: tuple, output_shape: int, model_type: str = "uniform", num_layers: int = 3):
        self.__name = name or self.__class__.__name__
        self.__input_shape = input_shape
        self.__output_shape = output_shape
        self.__config_data = CONFIG_DATA[f"{self.__class__.__name__}"]
        self.__model_type = model_type
        self.__num_layers = num_layers

        # print(self.__config_data)

        self.__model = None
        self.__build_model()

    def __build_model(self):
        self.__model = keras.models.Sequential(name=self.__name, trainable=True)
        self.__model.add(keras.layers.Input(shape=self.__input_shape))

        layer_info = self.__config_data['layers']
        conv_layer = layer_info['conv-layer']
        pooling_layer = layer_info['pooling-layer']
        
        conv_layer_class = getattr(keras.layers,conv_layer['type'])
        conv_layer_params = conv_layer['params']
        conv_filters = conv_layer_params['filters']
        conv_activation =conv_layer_params['activation']
        conv_kernel_size = conv_layer_params['kernel-size']

        pooling_layer_class = getattr(keras.layers, pooling_layer['type'])
        pooling_layer_params = pooling_layer['params']
        pooling_size = pooling_layer_params['pool-size']

        if self.__model_type.lower() == "uniform":
            for _ in range(self.__num_layers):
                self.__model.add(conv_layer_class(conv_filters, activation=conv_activation,kernel_size=conv_kernel_size))
                self.__model.add(pooling_layer_class(pool_size=pooling_size))
        
        elif self.__model_type.lower() == "incremental":
            for idx in range(1, self.__num_layers + 1):
                self.__model.add(conv_layer_class(conv_filters * idx, activation=conv_activation,kernel_size=conv_kernel_size))
                self.__model.add(pooling_layer_class(pool_size=pooling_size))
        
        elif self.__model_type.lower() == "decremental":
            for idx in range(self.__num_layers, 0, -1):
                self.__model.add(conv_layer_class(conv_filters * idx, activation=conv_activation,kernel_size=conv_kernel_size))
                self.__model.add(pooling_layer_class(pool_size=pooling_size))
        
        else:
            raise ValueError(f"Invalid model_type '{self.__model_type}'. Choose from 'uniform', 'incremental', 'decremental'.")
        

        output_config = self.__config_data['output-layer']
        flatten_layer = output_config['flatten-layer']
        flatten_layer_class = getattr(keras.layers,flatten_layer['type']) 

        dense_layer = output_config['dense-layer']
        dense_layer_class = getattr(keras.layers,dense_layer['type'])
        dense_params = dense_layer['params']
        dense_activation = dense_params['activation']

        self.__model.add(flatten_layer_class())
        self.__model.add(dense_layer_class(self.__output_shape, activation=dense_activation))

        self.__model.compile(optimizer=self.__config_data['optimizer'], loss=self.__config_data['loss'], metrics=['accuracy'])

    def summary(self):
        return self.__model.summary()
    
    def train(self, x_train: Any, y_train: Any, epchos: int = 10, verbose:int = 1):
        history = self.__model.fit(x_train,y_train, epochs=epchos, verbose=verbose)
        return history
    
    def predict(self, x_input: Any, verbose: int = 0):
        predictions = self.__model.predict(x_input, verbose=verbose)
        return predictions

    def evaluate(self, x_test: Any, y_test: Any, verbose: int = 1):
        evaluation = self.__model.evaluate(x_test, y_test, verbose=verbose)
        return evaluation

    def save(self):
        self.__model.save(f'./ml-models/{self.__name.lower()}_xs_image_classification_network.keras')

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

class LargeImageClassificationNetwork:
    """Builds and manages a large convolutional neural network for image classification tasks."""

    def __init__(self, name: str, input_shape: tuple, output_shape: int, model_type: str = "uniform", num_layers: int = 3):
        self.__name = name or self.__class__.__name__
        self.__input_shape = input_shape
        self.__output_shape = output_shape
        self.__config_data = CONFIG_DATA[f"{self.__class__.__name__}"]
        self.__model_type = model_type
        self.__num_layers = num_layers

        # print(self.__config_data)

        self.__model = None
        self.__build_model()

    def __build_model(self):
        self.__model = keras.models.Sequential(name=self.__name, trainable=True)
        self.__model.add(keras.layers.Input(shape=self.__input_shape))

        layer_info = self.__config_data['layers']
        conv_layer = layer_info['conv-layer']
        pooling_layer = layer_info['pooling-layer']
        
        conv_layer_class = getattr(keras.layers,conv_layer['type'])
        conv_layer_params = conv_layer['params']
        conv_filters = conv_layer_params['filters']
        conv_activation =conv_layer_params['activation']
        conv_kernel_size = conv_layer_params['kernel-size']

        pooling_layer_class = getattr(keras.layers, pooling_layer['type'])
        pooling_layer_params = pooling_layer['params']
        pooling_size = pooling_layer_params['pool-size']

        if self.__model_type.lower() == "uniform":
            for _ in range(self.__num_layers):
                self.__model.add(conv_layer_class(conv_filters, activation=conv_activation,kernel_size=conv_kernel_size))
                self.__model.add(pooling_layer_class(pool_size=pooling_size))
        
        elif self.__model_type.lower() == "incremental":
            for idx in range(1, self.__num_layers + 1):
                self.__model.add(conv_layer_class(conv_filters * idx, activation=conv_activation,kernel_size=conv_kernel_size))
                self.__model.add(pooling_layer_class(pool_size=pooling_size))
        
        elif self.__model_type.lower() == "decremental":
            for idx in range(self.__num_layers, 0, -1):
                self.__model.add(conv_layer_class(conv_filters * idx, activation=conv_activation,kernel_size=conv_kernel_size))
                self.__model.add(pooling_layer_class(pool_size=pooling_size))
        
        else:
            raise ValueError(f"Invalid model_type '{self.__model_type}'. Choose from 'uniform', 'incremental', 'decremental'.")
        

        output_config = self.__config_data['output-layer']
        flatten_layer = output_config['flatten-layer']
        flatten_layer_class = getattr(keras.layers,flatten_layer['type']) 

        dense_layer = output_config['dense-layer']
        dense_layer_class = getattr(keras.layers,dense_layer['type'])
        dense_params = dense_layer['params']
        dense_activation = dense_params['activation']

        self.__model.add(flatten_layer_class())
        self.__model.add(dense_layer_class(self.__output_shape, activation=dense_activation))

        self.__model.compile(optimizer=self.__config_data['optimizer'], loss=self.__config_data['loss'], metrics=['accuracy'])

    def summary(self):
        return self.__model.summary()
    
    def train(self, x_train: Any, y_train: Any, epchos: int = 10, verbose:int = 1):
        history = self.__model.fit(x_train,y_train, epochs=epchos, verbose=verbose)
        return history
    
    def predict(self, x_input: Any, verbose: int = 0):
        predictions = self.__model.predict(x_input, verbose=verbose)
        return predictions

    def evaluate(self, x_test: Any, y_test: Any, verbose: int = 1):
        evaluation = self.__model.evaluate(x_test, y_test, verbose=verbose)
        return evaluation

    def save(self):
        self.__model.save(f'./ml-models/{self.__name.lower()}_xs_image_classification_network.keras')

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

    
class XLargeImageClassificationNetwork:
    """Builds and manages an extra large convolutional neural network for image classification tasks."""

    def __init__(self, name: str, input_shape: tuple, output_shape: int, model_type: str = "uniform", num_layers: int = 3):
        self.__name = name or self.__class__.__name__
        self.__input_shape = input_shape
        self.__output_shape = output_shape
        self.__config_data = CONFIG_DATA[f"{self.__class__.__name__}"]
        self.__model_type = model_type
        self.__num_layers = num_layers

        # print(self.__config_data)

        self.__model = None
        self.__build_model()

    def __build_model(self):
        self.__model = keras.models.Sequential(name=self.__name, trainable=True)
        self.__model.add(keras.layers.Input(shape=self.__input_shape))

        layer_info = self.__config_data['layers']
        conv_layer = layer_info['conv-layer']
        pooling_layer = layer_info['pooling-layer']
        
        conv_layer_class = getattr(keras.layers,conv_layer['type'])
        conv_layer_params = conv_layer['params']
        conv_filters = conv_layer_params['filters']
        conv_activation =conv_layer_params['activation']
        conv_kernel_size = conv_layer_params['kernel-size']

        pooling_layer_class = getattr(keras.layers, pooling_layer['type'])
        pooling_layer_params = pooling_layer['params']
        pooling_size = pooling_layer_params['pool-size']

        if self.__model_type.lower() == "uniform":
            for _ in range(self.__num_layers):
                self.__model.add(conv_layer_class(conv_filters, activation=conv_activation,kernel_size=conv_kernel_size))
                self.__model.add(pooling_layer_class(pool_size=pooling_size))
        
        elif self.__model_type.lower() == "incremental":
            for idx in range(1, self.__num_layers + 1):
                self.__model.add(conv_layer_class(conv_filters * idx, activation=conv_activation,kernel_size=conv_kernel_size))
                self.__model.add(pooling_layer_class(pool_size=pooling_size))
        
        elif self.__model_type.lower() == "decremental":
            for idx in range(self.__num_layers, 0, -1):
                self.__model.add(conv_layer_class(conv_filters * idx, activation=conv_activation,kernel_size=conv_kernel_size))
                self.__model.add(pooling_layer_class(pool_size=pooling_size))
        
        else:
            raise ValueError(f"Invalid model_type '{self.__model_type}'. Choose from 'uniform', 'incremental', 'decremental'.")
        

        output_config = self.__config_data['output-layer']
        flatten_layer = output_config['flatten-layer']
        flatten_layer_class = getattr(keras.layers,flatten_layer['type']) 

        dense_layer = output_config['dense-layer']
        dense_layer_class = getattr(keras.layers,dense_layer['type'])
        dense_params = dense_layer['params']
        dense_activation = dense_params['activation']

        self.__model.add(flatten_layer_class())
        self.__model.add(dense_layer_class(self.__output_shape, activation=dense_activation))

        self.__model.compile(optimizer=self.__config_data['optimizer'], loss=self.__config_data['loss'], metrics=['accuracy'])

    def summary(self):
        return self.__model.summary()
    
    def train(self, x_train: Any, y_train: Any, epchos: int = 10, verbose:int = 1):
        history = self.__model.fit(x_train,y_train, epochs=epchos, verbose=verbose)
        return history
    
    def predict(self, x_input: Any, verbose: int = 0):
        predictions = self.__model.predict(x_input, verbose=verbose)
        return predictions

    def evaluate(self, x_test: Any, y_test: Any, verbose: int = 1):
        evaluation = self.__model.evaluate(x_test, y_test, verbose=verbose)
        return evaluation

    def save(self):
        self.__model.save(f'./ml-models/{self.__name.lower()}_xs_image_classification_network.keras')

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

    




if __name__ == "__main__":
    inn = SmallImageClassificationNetwork("asd",(24,24,3),3) 
    inn.summary()