import cv2
import numpy as np
from sklearn.cluster import KMeans

img = cv2.imread("./mond.jpg")

clt = KMeans(n_clusters=5)               # Set K-Means Cluster Alogorithm
clt_result = clt.fit(img.reshape(-1, 3)) # Run K-Means Cluster Alogorithm

# Color_set Varaiable has main color as format : (R, G, B)
color_set = np.array(clt.cluster_centers_, dtype=np.int16)

# Sort our color_set for consistent indexing
color_set_sorted_index = np.argsort(color_set[:, 0])

# Sort Alogorithm
color_set_sorted = np.empty_like(color_set)
for i in range(len(color_set)):
    color_set_sorted[i] = color_set[color_set_sorted_index[i]]

# Copy color_set_sorted as our color_set again
color_set = color_set_sorted
    
# To explain the main color, visualize the color_set and save as image.
width=img.shape[1]
color_set_visualize = np.zeros((50, width, 3), np.uint8)
steps = width/color_set.shape[0]
for idx, centers in enumerate(color_set): 
    color_set_visualize[:, int(idx*steps):(int((idx+1)*steps)), :] = centers

cv2.imwrite("color_set_visualized.png", color_set_visualize)

# Image Pre-Processing Before KMeans Alogirhtm
for i in range(img.shape[1]):
    for j in range(img.shape[0]):
        # Calculate the pixel value to compare our color_set
        distance = color_set - img[j][i]
        abs_distance = np.abs(distance)
        sum_distance = np.sum(abs_distance, axis=1)
        
        index = np.argmin(sum_distance)
                
        # Replace origin image to our simplified color_set
        img[j][i] = color_set[index]
        
# Replace selected color_set to other color_set
target_index = 1 # Target color index in color_set.
for i in range(img.shape[1]):
    for j in range(img.shape[0]):
        if list(img[j][i]) == list(color_set[target_index]):
            img[j][i] = [133, 215, 133] # Set Your Color

cv2.imshow("Color Replacement", img)
cv2.imwrite("color_modified.png", img)

cv2.waitKey(0)    
cv2.destroyAllWindows()