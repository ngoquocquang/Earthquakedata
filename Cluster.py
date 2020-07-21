import numpy as numpy
import scipy as scipy
from sklearn import cluster
import matplotlib.pyplot as plt
import pandas as pd

database=pd.read_csv('earthquake.csv')


def set2List(NumpyArray):
    list = []
    for item in NumpyArray:
        list.append(item.tolist())
    return list


def GenerateData():
    longitude = database[database['date']>='2015.01.01']['longitude'].values
    latitude = database[database['date']>='2015.01.01']['latitude'].values
    z = numpy.array([longitude, latitude]).T
    return z


def DBSCAN(Dataset, Epsilon,MinumumPoints,DistanceMethod = 'euclidean'):
#    Dataset is a mxn matrix, m is number of item and n is the dimension of data
    m,n=Dataset.shape
    Visited=numpy.zeros(m,'int')
    Type=numpy.zeros(m)
#   -1 noise, outlier
#    0 border
#    1 core
    ClustersList=[]
    Cluster=[]
    PointClusterNumber=numpy.zeros(m)
    PointClusterNumberIndex=1
    PointNeighbors=[]
    #chuyển khoảng cách vecto về dạng vuông
    DistanceMatrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(Dataset, DistanceMethod))
    for i in range(m):
        if Visited[i]==0:
            Visited[i]=1
            #nếu khoảng cách nhỏ hơn epsilon thì trả về giá trị 0
            PointNeighbors=numpy.where(DistanceMatrix[i]<Epsilon)[0]
            if len(PointNeighbors)<MinumumPoints:
                Type[i]=-1
            else:
                for k in range(len(Cluster)):
                    Cluster.pop()
                Cluster.append(i)
                PointClusterNumber[i]=PointClusterNumberIndex
                
                
                PointNeighbors=set2List(PointNeighbors)    
                ExpandClsuter(Dataset[i], PointNeighbors,Cluster,MinumumPoints,Epsilon,Visited,DistanceMatrix,PointClusterNumber,PointClusterNumberIndex  )
                Cluster.extend(PointNeighbors[:])
                ClustersList.extend(Cluster[:])
                PointClusterNumberIndex=PointClusterNumberIndex+1
                ClustersList=list(set(ClustersList))
    print(PointClusterNumberIndex - 1)

                    
    return PointClusterNumber, ClustersList 



def ExpandClsuter(PointToExapnd, PointNeighbors,Cluster,MinumumPoints,Epsilon,Visited,DistanceMatrix,PointClusterNumber,PointClusterNumberIndex  ):
    Neighbors=[]

    for i in PointNeighbors:
        if Visited[i]==0:
            Visited[i]=1
            Neighbors=numpy.where(DistanceMatrix[i]<Epsilon)[0]
            if len(Neighbors)>=MinumumPoints:
#                Neighbors merge with PointNeighbors
                for j in Neighbors:
                    try:
                        PointNeighbors.index(j)
                    except ValueError:
                        PointNeighbors.append(j)
                    
        if PointClusterNumber[i]==0:
            Cluster.append(i)
            PointClusterNumber[i]=PointClusterNumberIndex
    return

#Generating some data with normal distribution at 
#(0,0)
#(8,8)
#(12,0)
#(15,15)
Data=GenerateData()

#Adding some noise with uniform distribution 
#X between [-3,17],
#Y between [-3,17]



Epsilon= 0.5
MinumumPoints=20
result, clusterpoint =DBSCAN(Data,Epsilon,MinumumPoints)


fig = plt.figure()
ax1=fig.add_subplot(1,1,1) #row, column, figure number

ax1.scatter(Data[:,0],Data[:,1], color='green', alpha =  0.5)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('2015 - 2017')

#printed numbers are cluster numbers
print(result)
#print "Noisy_Data"
#print Noisy_Data.shape
#print Noisy_Data

for i in clusterpoint:
    ax1.scatter(Data[i][0], Data[i][1], color='orange' , alpha =  0.5)
      
plt.show()