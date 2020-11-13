import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors

class RestaurantRecommendationSystem:
    def __init__(self,dataset):
        self.data = pd.read_csv(dataset)
        self.dataset = self.data.copy()
        self.data_cleaning()

    def data_cleaning(self):
        del self.data['url']
        del self.data['phone']
        del self.data['address']

        self.data.isnull().sum()

        self.data['rate'] = self.data['rate'].replace('NEW',np.NaN)
        self.data['rate'] = self.data['rate'].replace('-',np.NaN)
        self.data=self.data.rename(columns={'approx_cost(for two people)':'cost','listed_in(type)':'type','listed_in(city)':'city'})

        self.X=self.data.copy()
        self.X.online_order=self.X.online_order.apply(lambda x: '1' if str(x)=='Yes' else '0')
        self.X.book_table=self.X.book_table.apply(lambda x: '1' if str(x)=='Yes' else '0')

        self.X.rate=self.X.rate.astype(str)
        self.X.rate=self.X.rate.apply(lambda x : x.replace('/5',''))
        self.X.rate=self.X.rate.astype(float)

        self.X.cost=self.X.cost.astype(str)
        self.X.cost=self.X.cost.apply(lambda y : y.replace(',',''))
        self.X.cost=self.X.cost.astype(float)

        self.X.online_order=self.X.online_order.astype(float)
        self.X.book_table=self.X.book_table.astype(float)
        self.X.votes=self.X.votes.astype(float)

        self.X_del=self.X.copy()
        self.X_del.dropna(how='any',inplace=True)

        self.X_del.drop_duplicates(keep='first',inplace=True)

    def data_visualisation(self):

        #Pie chart for understanding persentage of people from each location
        s=set()
        d=dict()
        for i in self.X_del.city:
            if i not in s:
                s.add(i)
                d.update({i:1})
            else:
                d[i]+=1

        labels = d.keys()
        sizes = d.values()
        colors=[]

        for x in labels:
            rgb = (random.random(), random.random(), random.random())
            colors.append(rgb)

        plt.pie(sizes, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True, startangle=140)
        plt.title("Understanding percentage of people from each location\n\n")
        plt.axis('equal')
        plt.show()


        # Pie chart for understanding percentage of restaurants offering online delivery
        s=set()
        d=dict()

        for i in self.X_del.online_order:
            if i not in s:
                s.add(i)
                d.update({i:1})
            else:
                d[i]+=1

        labels = d.keys()
        sizes = d.values()
        colors=[]

        for x in labels:
            rgb = (random.random(), random.random(), random.random())
            colors.append(rgb)

        plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
        plt.title("Understanding the type category from each location")
        plt.axis('equal')
        plt.show()


        #Bar chart showing avg cost of food for each area of the city.
        s=set()
        d=dict()
        
        for i in self.X_del.city:
            if i not in s:
                s.add(i)
                d.update({i:1})
            else:
                d[i]+=1

        d_cost=dict()

        for i in s:
            d_cost.update({i:0})

        for i in self.X_del.index:
            d_cost[self.X_del.city[i]]+=self.X_del.cost[i]

        for i in s:
            d_cost[i]=d_cost[i]/d[i]

        plt.title("Average cost vs Location")
        plt.ylabel('Average cost')
        plt.bar(d.keys(),height=d_cost.values())
        plt.xticks(rotation=90)
        plt.show()


        #Scatter plot for votes vs cost
        x = self.X_del.cost
        y = self.X_del.votes

        plt.title('Scatter plot for cost vs votes')
        plt.scatter(x,y)
        plt.show()


        #Histogram for cost
        x = self.X_del.cost

        plt.title('Histogram plot for cost')
        plt.hist(x,bins=50)
        plt.show()


        #Bar chart showing avg no of cuisines offered in each area
        s=set()
        d=dict()

        count_d=dict()

        for i,j in zip(self.X_del.city,self.X_del.cuisines):
            if i not in s:
                s.add(i)
                count_d.update({i:1})
                d.update({i:len(j.split(', '))})
            else:
                count_d[i]+=1
                d[i]+=len(j.split(', '))

        for i,j in zip(d,count_d):
            d[i]=d[i]/count_d[i]

        plt.title("Average no of cuisines offered vs Location")
        plt.ylabel('Average no of cuisines')
        plt.bar(d.keys(),height=d.values())
        plt.xticks(rotation=90)
        plt.show()


    def predict(self):
        cluster_with_name=self.X_del.iloc[:,[9,3,0]]
        cos=input("Enter The Cost :")
        rat=input("Enter The Rating 1-5 :")
        input_location=input("Enter the location :")
        filtered_data=self.X_del.loc[self.X_del['city'] == input_location]
        new_row = {'cost':cos, 'rate':rat,'name':"user"}
        cluster_with_name = cluster_with_name.append(new_row, ignore_index=True)

        checkCluster=filtered_data.iloc[:,[9,3]]

        new_row = {'cost':cos, 'rate':rat}
        checkCluster = checkCluster.append(new_row, ignore_index=True)
        #here the costumer gives 600 as expected cost and 3.5 as expected rating soo adding it to the last index
        # checkCluster
        # cluster_with_name

        #need to filter according to the given constraint then use clustering

        
        dendrogram = sch.dendrogram(sch.linkage(checkCluster, method = 'ward'))
        plt.title('Dendrogram')
        plt.xlabel('Customers')
        plt.ylabel('Euclidean distances')
        plt.show()

        
        hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
        y_hc = hc.fit_predict(checkCluster)
        # y_hc
        X=cluster_with_name.values
        #print(X)
        plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
        plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
        plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
        plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
        plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
        plt.title('Clusters of Restuarants')
        plt.xlabel('Cost $')
        plt.ylabel('Rating(1-5)')
        plt.legend()
        plt.show()

        #to select a particular cluster corresponding to the users input choice
        #add to the last the attributes selected by the user then check for its position here

        selected_cluster_no=y_hc[-1]
        # print(selected_cluster_no)  #cluster no=output +1

        X=X[y_hc == selected_cluster_no,]#numpy nd array
        X[:,-3:-1]

        
        nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(X[:,-3:-1])
        distances, indices = nbrs.kneighbors(X[:,-3:-1])

        # indices

        # distances

        # indices[-1]

        #The top 5 recommended restaurants
        count=0
        for i in indices[-1]:
            if(X[i][2]=="user"):
                continue
            count=count+1
            print(X[i],"\n",count," Name:",X[i][2],"\n")

        # distances[-1]

    
r1=RestaurantRecommendationSystem(dataset='mini.csv')
#r1.data_visualisation()
r1.predict()