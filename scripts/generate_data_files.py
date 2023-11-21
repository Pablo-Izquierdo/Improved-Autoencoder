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

DATASET_DIR_PATH = "/storage/"
NORMAL_DATA_DIR_NAME = "normal_all"
ABNORMAL_DATA_DIR_NAME = "abnormal_all"
NORMAL_DATA_DIR_PATH = DATASET_DIR_PATH+NORMAL_DATA_DIR_NAME+"/"
ABNORMAL_DATA_DIR_PATH = DATASET_DIR_PATH+ABNORMAL_DATA_DIR_NAME+"/"
SELECTIVE_IMAGES_TRAIN_FILEPATH = DATASET_DIR_PATH+"train.txt"
SELECTIVE_IMAGES_TEST_FILEPATH = DATASET_DIR_PATH+"test.txt"

NORMAL_IDENTIFIER= 1
ABNORMAL_IDENTIFIER= 0

RESULT_FILES_DIRECTORY = '/srv/Improved-Autoencoder/dataset/data_files/'
pwd = os.getcwd()
print(pwd)

try:
    os.mkdir(RESULT_FILES_DIRECTORY)
except:
    print("directory already exists")

def read_file_adding_data(filepath):

    dict = {}
    df = pd.DataFrame(columns=['path', 'clase'])

    with open(filepath, 'r') as f:
        for l in f:
            splited = l.split('*')
            name = splited[0]
            img_class = int(splited[1])

            dict[str(name)]=img_class #add img to dir

            if img_class == NORMAL_IDENTIFIER:
                full_path = NORMAL_DATA_DIR_PATH+name
            elif img_class == ABNORMAL_IDENTIFIER:
                full_path = ABNORMAL_DATA_DIR_PATH+name
            else:
                raise RuntimeError("class identifier not recognize")

            df = df._append(pd.DataFrame([[full_path,int(img_class)]], columns=['path', 'clase']), ignore_index=True)
        f.close()

    return dict, df

#Funcion que genera un dict con los datos que deben utilizarse en train o test de forma obligatoria
def get_selective_data():

    all_selective_data = {}
    df_test = pd.DataFrame(columns=['path', 'clase'])
    df_train = pd.DataFrame(columns=['path', 'clase'])


    if os.path.isfile(SELECTIVE_IMAGES_TRAIN_FILEPATH):
        dict_train, df_train = read_file_adding_data(SELECTIVE_IMAGES_TRAIN_FILEPATH)
        all_selective_data.update(dict_train)
    elif os.path.isfile(SELECTIVE_IMAGES_TEST_FILEPATH):
        dict_test, df_test = read_file_adding_data(SELECTIVE_IMAGES_TEST_FILEPATH)
        all_selective_data.update(dict_test)

    return all_selective_data, df_train, df_test
    
# funcion que genera el fichero dataset.txt con las carpetas de imagenes y un dataframe con path y etiqueta
def generate_dataset_file(imageDirectory, num_abnormal, dict_selective_data):

    count_abnormal = 0

    # VALUES(date, user, picture, hash, location, idfruta, idvariedad, tamaño, luz, plano, angulo, plato, superficie);
    with open(RESULT_FILES_DIRECTORY+'dataset.txt', 'w') as fw: #Escribo en fichero dataset
        df = pd.DataFrame(columns=['path', 'clase'])
        type = ABNORMAL_IDENTIFIER
        for img_class in [NORMAL_DATA_DIR_NAME, ABNORMAL_DATA_DIR_NAME]: # Para cada clase (normal / abnormal)
            img_class_Directory = imageDirectory + img_class + "/"
            #print(img_class_Directory)

            if img_class == NORMAL_DATA_DIR_NAME: # normal = 1 / abnormal = 0
                type = NORMAL_IDENTIFIER
            else:
                type = ABNORMAL_IDENTIFIER

            for img_name in os.listdir(img_class_Directory):
                if img_name not in dict_selective_data:
                    if (type == ABNORMAL_IDENTIFIER):
                        count_abnormal = count_abnormal + 1

                    if (count_abnormal <= num_abnormal) or (type == NORMAL_IDENTIFIER):
                        #Incluimos las imagenes del directorio en fichero y dataset
                        full_path = img_class_Directory + img_name
                        #print(full_path)
                        df = df._append(pd.DataFrame([[full_path,int(type)]], columns=['path', 'clase']), ignore_index=True)
                
                        string = str(full_path) + '*' + str(type)+'\n'
                        fw.write(string)
                
    fw.close()

    return df

#Escribo en el fichero el path y etiqueta
def write_file(writefile, X, y):
    
    print("Writing "+ writefile)
    with open(RESULT_FILES_DIRECTORY+writefile, 'w') as fw: #Escribo en fichero Train
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
def gen_data(images_dir, num_abnormal):

    #get data that has to be at one specifict group (train or test)
    dict_selective_data, df_train, df_test = get_selective_data()
    
    #Genero dataset.txt y obtengo df [path, size]
    df = generate_dataset_file(images_dir, num_abnormal, dict_selective_data)
    print(df)

    #Dristribuir en train, test
    print(Counter(df.clase))
    X_train, X_test, y_train, y_test = train_test_split(df.path, df.clase, test_size=0.20, random_state=1, stratify=df.clase)
    print("Train:", Counter(y_train))
    print("Test:",Counter(y_test))
    #Train
    write_file('train.txt', pd.concat([X_train, df_train['path']]), pd.concat([y_train, df_train['clase']]))
    #Test
    write_file('test.txt', pd.concat([X_test,df_test['path']]), pd.concat([y_test,df_test['clase']]))
    
    
gen_data(DATASET_DIR_PATH, 4574)