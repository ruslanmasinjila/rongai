# VERSION 3 
# imports
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#####################################################################################
# LOADING AND CLEANING OF RONGAI DATA
#####################################################################################

# load rongai coordinates
rongai_data = pd.read_csv("rongai_data.txt", delimiter="\t")

# get latitudes and longitudes from rongai coordinates and add 0 sanitation (expand the data to 3D)
lat_long_rongai = rongai_data[['latitude', 'longitude']]
lat_long_rongai['sanitation rates'] = 0

#####################################################################################
# LOADING AND CLEANING OF SANITATION DATA
#####################################################################################

# Load sanitation coordinates
sanitation_data = pd.read_csv("sanitation_data.csv")

# Get the GPS coordinates for sanitation_data
lat_long_sanitation= sanitation_data[['GPS coordinates']]

# Split the GPS coordinates for sanitation areas to latitudes and longitudes
lat_long_sanitation[['latitude','longitude']] = lat_long_sanitation['GPS coordinates'].str.split(',',expand=True)

# Drop the GPS coordinates (combined) after spliting into latitudes and longitudes
lat_long_sanitation= lat_long_sanitation.drop(['GPS coordinates'], axis=1)

# Add the sanitation rates to sanitation areas
lat_long_sanitation['sanitation rates'] = sanitation_data['Rate the sanitary state of the identified feature']

# Take reciprocal of the sanitation rates
lat_long_sanitation['sanitation rates'] = 1/lat_long_sanitation['sanitation rates']

# Convert Strings to floats
lat_long_sanitation['latitude'] =  lat_long_sanitation['latitude'].astype(float).fillna(0.0)
lat_long_sanitation['longitude'] = lat_long_sanitation['longitude'].astype(float).fillna(0.0)

#####################################################################################
# PAIRWISE EUCLIDEAN DISTANCES
#####################################################################################

# Convert dataframes to numpy
np_lat_long_rongai = lat_long_rongai.to_numpy()
np_lat_long_sanitation = lat_long_sanitation.to_numpy()

# Calculate pairwise euclidean distances
eu_distances = euclidean_distances(np_lat_long_rongai,np_lat_long_sanitation)
avg_eu_distances = np.mean(eu_distances, axis=1)

#####################################################################################
# ATTACHING EUCLIDEAN DISTANCES TO RONGAI LATITUDES AND LONGITUDES
#####################################################################################
np_lat_long_rongai[:,2] = avg_eu_distances
np_lat_long_eu_rongai = np_lat_long_rongai

#####################################################################################
# STANDARDIZATION OF THE DATA
#####################################################################################
np_lat_long_eu_rongai_standardized = (np_lat_long_eu_rongai - np.mean(np_lat_long_eu_rongai)) / np.std(np_lat_long_eu_rongai)


#####################################################################################
# BEGIN K-MEANS
#####################################################################################
kmeans = KMeans(n_clusters= 6)
label = kmeans.fit_predict(np_lat_long_eu_rongai_standardized) # USE STANDARDIZED DATA HERE
u_labels = np.unique(label)


#####################################################################################
# 3D PLOT
#####################################################################################
fig = plt.figure()
ax = plt.axes(projection='3d')
for i in u_labels:
    ax.scatter(np_lat_long_eu_rongai[label == i , 0] , np_lat_long_eu_rongai[label == i , 1], np_lat_long_eu_rongai[label == i , 2] , label = i) # USE NON-STANDARDIZED DATA HERE
plt.legend()
plt.show()

'''
###>>>>>>>>>>>>>>>>>>>>>>>>>>> DEBUGGING ZONE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#####################################################################################
# FIND OPTIMUM NUMBER OF CLUSTERS (REMOVE THIS PART LATER)
#####################################################################################
cost =[] 
for i in range(1, 11): 
    KM = KMeans(n_clusters = i, max_iter = 500) 
    KM.fit(np_lat_long_eu_rongai_standardized) # USE STANDARDIZED DATA HERE
      
    # calculates squared error 
    # for the clustered points 
    cost.append(KM.inertia_)      
  
# plot the cost against K values 
plt.plot(range(1, 11), cost, color ='g', linewidth ='3') 
plt.xlabel("Value of K") 
plt.ylabel("Sqaured Error (Cost)") 
plt.show() # clear the plot 
'''


'''
ax = plt.gca()
ax.scatter(np_lat_long_rongai[:,0],np_lat_long_rongai[:,1], color="b")
ax.scatter(np_lat_long_sanitation[:,0],np_lat_long_sanitation[:,1], color="r")
plt.show()
'''


