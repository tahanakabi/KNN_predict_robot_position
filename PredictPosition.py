import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor as knn
import time

## This script uses a set of positions (x,y) of a Robot to predict the next 60 positions.
## The problem is formulated as a regression plroblem using the last position coordinates and velocity
## We used a KNN model having x,y, x_velocity, y_velocity as inputs and x,y as outputs


def load_data(filename):

    x=[]
    y=[]
    with open(filename, 'r') as d:  # using the with keyword ensures files are closed properly
        for line in d.readlines():
            parts = line.split(',')  # change this to whatever the deliminator is
            x.append(float(parts[0]))
            y.append(float(parts[1].replace('\n','')))
    data=np.array([x,y])
    return data

def prepare_data(data):
    x_velocity=[]
    y_velocity=[]
    for index in range(data[:,:-1].shape[1]):
        x_velocity.append(data[0][index+1]-data[0][index])
        y_velocity.append(data[1][index+1]-data[1][index])
    data=data[:,1:]
    result = np.insert(data,2,x_velocity,axis=0)
    result = np.insert(result, 3, y_velocity, axis=0)
    return result

def normalise_data(data):
    max_x=np.max(data[0])
    max_y=np.max(data[1])
    min_x=np.min(data[0])
    min_y=np.min(data[1])
    data[0] = np.true_divide((data[0]-min_x),(max_x-min_x))
    data[1] = np.true_divide((data[1] - min_y) , (max_y - min_y))
    data[2] = np.true_divide(data[2] , (max_x - min_x))
    data[3] = np.true_divide(data[3],(max_y - min_y))
    return data, max_x,max_y,min_x,min_y

def denormalise_data(data,max_x,max_y,min_x,min_y):
    data[:,0] =  data[:,0] * (max_x - min_x) + min_x
    data[:,1] =  data[:,1] * (max_y - min_y) + min_y
    return data

def get_X_y(data,train_size):
    y=data[0:2,1:].T
    X=data[:,:-1].T
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=0)
    return X_train, X_test, y_train, y_test

def get_test_data(data, lenght):
    if (lenght==0):
        X = data[:, - 2:].T
        y=[]
    else:
        y = data[0:2, -lenght:].T
        X = data[:, -lenght-2:-lenght].T
    return X,y

def train_model(X,y):
    clf = knn(n_neighbors=30, weights='distance',n_jobs=1)
    clf.fit(X,y)
    return clf


def predict(model,data):
    data=data[-1:,:]
    new_data=data[:,0:2]
    for i in range(60):
        data = data[-1:, :]
        prediction=model.predict(data)
        new_data=np.append(new_data,prediction,axis=0)
        data=prepare_data(new_data.T).T
    return new_data[1:,:]


if __name__ == '__main__':
    t0=time.time()

    data=load_data('training_data.txt')
    data,max_x,max_y,min_x,min_y=normalise_data(prepare_data(data))
    X_train, X_test, y_train, y_test=get_X_y(data,0.9)
    clf=train_model(X_train,y_train)

    files = ['test01.txt', 'test02.txt', 'test03.txt', 'test04.txt', 'test05.txt', 'test06.txt', 'test07.txt',
             'test08.txt', 'test09.txt', 'test10.txt']

    for file in files:
        raw_data = load_data(file)
        data, max_x, max_y, min_x, min_y = normalise_data(prepare_data(raw_data))
        X_train, X_test, y_train, y_test = get_X_y(data, 0.9)
        clf.fit(X_train,y_train)
        X_test,y_test=get_test_data(data,0)
        prediction=predict(clf,X_test)
        predicted_positions=denormalise_data(prediction,max_x,max_y,min_x,min_y)
        predicted_positions=np.round(predicted_positions,0).astype(int)
        real_positions=np.array(get_test_data(prepare_data(raw_data),60)[1])
        distances=np.linalg.norm(real_positions-predicted_positions,axis=1)**2
        error=np.sqrt(np.sum(distances))
        print('> ', file , "Error : ", error)
        np.savetxt("predictions.txt", predicted_positions,fmt='%i', delimiter=",")

    print("> Total Time : ", time.time() - t0)


