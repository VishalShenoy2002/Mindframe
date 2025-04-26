from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np
import os

from typing import Union, List
from keras.utils import to_categorical

class DatasetProcessor:
    def __init__(self, filename: str, train_size: float, test_size:float):
        self.__filename = filename

        self.__test_size = test_size
        self.__train_size = train_size

        self.__data = None
        self. __x = None
        self.__y = None

        self.__x_train = None
        self.__x_test = None
        self.__y_train = None
        self.__y_test = None

        self.__x_train_vectorized = None
        self.__x_test_vectorized = None

        self.__y_train_encoded = None
        self.__y_test_encoded = None
        
    def load(self,input_columns: Union[List[str], str], output_column: str):

        if not os.path.exists(self.__filename):
            raise FileNotFoundError(f"{self.__filename} not found. Please provide a file which exists.")
        
        if not self.__filename.endswith('.csv'):
            raise ValueError(f'Please prove a file which is in CSV format.')
        
        self.__data = pd.read_csv(self.__filename)

        self.__x = self.__data[input_columns]
        self.__y = self.__data[output_column]

    
    def split(self):
        self.__x_train, self.__x_test, self.__y_train, self.__y_test = train_test_split(self.__x, self.__y, train_size=self.__train_size, test_size=self.__test_size, random_state=42)

    def vectorize_and_encode(self):
        
        vectorizer = TfidfVectorizer()
        encoder = LabelEncoder()

        if not self.__x_train or not self.__y_train:
            raise ValueError("The value of either the input or output training set is None. Please run the .split() function first.")
        
        if not self.__x_test or not self.__y_test:
            raise ValueError("The value of either the input or output training set is None. Please run the .split() function first.")
        

        self.__x_train_vectorized = vectorizer.fit_transform(self.__x_train)
        self.__x_test_vectorized = vectorizer.transform(self.__x_test)

        self.__y_train_encoded = to_categorical(encoder.fit_transform(self.__y_train))
        self.__y_test_encoded = to_categorical(encoder.fit_transform(self.__y_test))

    def get_segregated_data(self):
        return self.__x_train, self.__x_test, self.__y_train, self.__y_test


    def get_model_ready_data(self):
        return self.__x_train_vectorized, self.__x_test_vectorized, self.__y_train_encoded, self.__y_test_encoded


        