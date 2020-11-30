from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import math

# reading the dataset
data = pd.read_csv('zomato.csv')
dataset = data.copy()

p=None
ind=None

def prio(a):
    return abs((a[ind])-p)


#Data Cleaning

# deleting unnecessary data
del data['url']
del data['address']

data.isnull().sum()

# Replace Bogus terms with NaN values
data['rate'] = data['rate'].replace('NEW', np.NaN)
data['rate'] = data['rate'].replace('-', np.NaN)
data = data.rename(columns={'approx_cost(for two people)': 'cost', 'listed_in(type)': 'type','listed_in(city)': 'city'})

# Convert str to float
X = data.copy()
X.online_order = X.online_order.apply(lambda x: '1' if str(x) == 'Yes' else '0')
X.book_table = X.book_table.apply(lambda x: '1' if str(x) == 'Yes' else '0')

X.rate = X.rate.astype(str)
X.rate = X.rate.apply(lambda x: x.replace('/5', ''))
X.rate = X.rate.astype(float)

X.cost = X.cost.astype(str)
X.cost = X.cost.apply(lambda y: y.replace(',', ''))
X.cost = X.cost.astype(float)

X.online_order = X.online_order.astype(float)
X.book_table = X.book_table.astype(float)
X.votes = X.votes.astype(float)

# Now all value related columns are float type.
# Replace missing values by deleting missing values
X_del = X.copy()
X_del.dropna(how='any', inplace=True)

# Remove duplicates
X_del.drop_duplicates(subset='name', keep='first', inplace=True)

class RestaurantRecommendationSystem:
    def __init__(self):
        print()
        print("------------Restaurant Recommendation System------------\n\n")
        print("To Skip the any queries Enter \"skip\":\n")
        self.takeInput()
        self.fit()
        self.predict()
    
    def takeInput(self):
        global X_del

        # Input Location
        self.input_location = input("Enter Location :")
        if(self.input_location != "skip"):
            X_del = X_del.loc[X_del['city'] == self.input_location]

        if(len(X_del) == 0):
            print("Invalid location Entered")
            sys.exit()

        # Input Restaurant Type
        self.input_rest_type = input("Enter Restuarant Type :")
        if(self.input_rest_type != "skip"):
            X_del = X_del.loc[X_del['rest_type'] == self.input_rest_type]

        if(len(X_del) == 0):
            print("Invalid Restuarant Type Entered")
            sys.exit()

        # Input Required Cost
        self.cos = input("Enter Cost :")
        if(self.cos != "skip"):
            self.cos = float(self.cos)
        else:
            self.cos = X_del['cost'].mean()
            print(self.cos, "Is the Value Selected")

        # Input Required Rating
        self.rat = input("Enter The Rating 1-5 :")
        if(self.rat != "skip" and abs(float(self.rat)) <= 5):
            self.rat = abs(float(self.rat))
        else:
            self.rat = 5.0
        
        # Input the required priority
        self.prior = int(input("\nEnter 0 to give no priority , 1 to give priority to Cost and 2 to give priority to Rating: "))

        # The max number of recommendations that can be computed is found out by len(X)-1
        # Input the number of recommendations
        self.no_of_recomendations = int(input("Enter the no. of Recommendations :"))
    
    def fit(self):
        global X_del

        # Extracting the required columns for the model
        cluster_with_name = X_del.iloc[:, [10, 3, 0, 5]]

        # Adding the new cost,rating and user at the end of the dataset
        new_row = {'cost': self.cos, 'rate': self.rat, 'name': "user"}
        cluster_with_name = cluster_with_name.append(new_row, ignore_index=True)

        checkCluster = X_del.iloc[:, [10, 3]]

        new_row = {'cost': self.cos, 'rate': self.rat}
        checkCluster = checkCluster.append(new_row, ignore_index=True)

        #Visualizing Dendogram
        dendrogram = sch.dendrogram(sch.linkage(checkCluster, method='ward'))
        plt.title('Dendogram')
        plt.xlabel('Customers')
        plt.ylabel('Euclidean distances')
        plt.show()

        # Code for Clustering if the number of restaurants in the dataset can be clustered
        if(len(checkCluster) > 5):
            hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
            y_hc = hc.fit_predict(checkCluster)

            self.X = cluster_with_name.values
            plt.scatter(self.X[y_hc == 0, 0], self.X[y_hc == 0, 1],
                        s=100, c='red', label='Cluster 1')
            plt.scatter(self.X[y_hc == 1, 0], self.X[y_hc == 1, 1],
                        s=100, c='blue', label='Cluster 2')
            plt.scatter(self.X[y_hc == 2, 0], self.X[y_hc == 2, 1],
                        s=100, c='green', label='Cluster 3')
            plt.scatter(self.X[y_hc == 3, 0], self.X[y_hc == 3, 1],
                        s=100, c='cyan', label='Cluster 4')
            plt.scatter(self.X[y_hc == 4, 0], self.X[y_hc == 4, 1],
                        s=100, c='magenta', label='Cluster 5')
            plt.title('Clusters of Restuarants')
            plt.xlabel('Cost $')
            plt.ylabel('Rating(1-5)')
            plt.legend()
            plt.show()
            selected_cluster_no = y_hc[-1]
            self.single_cluster_X = self.X[y_hc == selected_cluster_no, ]  # numpy nd array
        else:
            print("Since the no of items in the filter dataset is less, the clustering is not done")
            self.X = cluster_with_name.values
            self.single_cluster_X = self.X


    def predict(self):

        global p
        global ind

        # Model exits if the number of recommendations is more than what si there in the dataser
        if(self.no_of_recomendations >= len(self.X)):
            print("These no. of recommendations cannot pe processed")
            sys.exit()

        # If the number of recommendations is lesser than what is present in the cluster select the cluster for K nearest neighbours
        if(self.no_of_recomendations < len(self.single_cluster_X)):
            self.X = self.single_cluster_X

        # Scaling the cost and the rating columns
        sc = StandardScaler()
        self.X[:, -4:-2] = sc.fit_transform(self.X[:, -4:-2])

        # Appling K nearest neighbours to the scaled data
        nbrs = NearestNeighbors(n_neighbors=self.no_of_recomendations+1,algorithm='auto').fit(self.X[:, -4:-2])
        distances, indices = nbrs.kneighbors(self.X[:, -4:-2])
        self.X[:, -4:-2] = sc.inverse_transform(self.X[:, -4:-2])

        # Relevant cost or rating choosen as per priority of the user for sorting
        if(self.prior == 1):
            p = self.cos
            ind = 0
        if(self.prior == 2):
            p = self.rat
            ind = 1

        # Extract all the data of the K nearest neighbours in a list except the users
        count = 0
        l = []
        for i in indices[-1]:
            if(self.X[i][2] == "user"):
                continue
            l.append(self.X[i].tolist())

        # Sorting as per priority
        if(self.prior != 0):
            l = sorted(l, key=prio)


        # Displaying the restaurant as per the users requirement
        count = 0
        sum_cost = 0
        sum_rat = 0
        for i in l:
            count = count+1
            print("\n", count, "| Name:", i[2], "| Cost: ", i[0], "| Rating: ", i[1],"| Phone Number: ",i[3].split()[0],i[3].split()[1])
            sum_cost += pow(i[0]-self.cos, 2)
            sum_rat += pow(i[1]-self.rat, 2)

        print()
        print()
        print("Cost Error in Rupees: ", math.sqrt(sum_cost/count))
        print("Rating Error: ", math.sqrt(sum_rat/self.rat))
        print()

rrs=RestaurantRecommendationSystem()
    
            
