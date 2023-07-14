from enum import Enum
from datetime import datetime
import os
import hashlib
import multiprocessing
from multiprocessing import Pool
import pandas as pd
from sklearn.model_selection import train_test_split
from skimage import io

from collections import Counter
import random as r

FILES_DIRECTORY = '../dataset/data_files/'
pwd = os.getcwd()
print(pwd)
# funcion que genera el chisero dataset.txt con las carpetas de imagenes y un dataframe con path y bin_size
def generate_dataset_file(imageDirectory):

    # VALUES(date, user, picture, hash, location, idfruta, idvariedad, tamaño, luz, plano, angulo, plato, superficie);
    with open(FILES_DIRECTORY+'dataset.txt', 'w') as fw: #Escribo en fichero dataset
        df = pd.DataFrame(columns=['path', 'clase'])
        type = 0
        for img_class in os.listdir(imageDirectory): # Para cada clase (normal / abnormal)
            img_class_Directory = imageDirectory + img_class + "/"
            #print(img_class_Directory)

            if img_class == 'normal': # normal = 1 / abnormal = 0
                type = 1
            else:
                type = 0

            for img_path in os.listdir(img_class_Directory):
                full_path = img_class_Directory + img_path
                #print(full_path)
                df = df._append(pd.DataFrame([[full_path,int(type)]], columns=['path', 'clase']), ignore_index=True)
        
                string = str(full_path) + '*' + str(type)+'\n'
                fw.write(string)
                
    fw.close()

    return df

#Escribo en el fichero el path y tamaño extraido del path
def write_file(writefile, X, y):
    
    print("Writing "+ writefile)
    with open(FILES_DIRECTORY+writefile, 'w') as fw: #Escribo en fichero Train
        for i in range(len(X)):
            file_path= X.iloc[i]
            type = y.iloc[i]
            #print(directory, tamaño)
            #checking if it is a file
            if not os.path.isfile(file_path):
                raise Exception("File Not found: " + str(file_path))

            try:
                _ = io.imread(file_path)
                string = str(file_path) + '*' + str(type)+'\n'
                fw.write(string)
            except Exception as e:
                print(file_path)
        fw.close()
        
 # Estratificamos datos y escribimos en fichero
def gen_data(images_dir):

    #Genero dataset.txt y obtengo df [path, size]
    df = generate_dataset_file(images_dir)
    print(df)
    #Dristribuir en train, test
    print(Counter(df.clase))
    X_train, X_test, y_train, y_test = train_test_split(df.path, df.clase, test_size=0.20, random_state=1, stratify=df.clase)
    print("Train:", Counter(y_train))
    print("Test:",Counter(y_test))
    #Train
    write_file('train.txt', X_train, y_train)
    #Test
    write_file('test.txt', X_test, y_test)
    
gen_data('/srv/wine_dataset/')