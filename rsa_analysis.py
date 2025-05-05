import numpy as np
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt


#loading beta coefficients into variable called beta
# ( width, height, depth, stimulus condition) array 4 dimension
beta = np.load('beta_coefficients.npy')

#load the atlas, indicates level of interest in the voxel 0-5, 0=ifnore
atlas = np.load('atlas.npy')


#this will load the total dimensions for each variable (beta and atlas)
#confirms that the arrays makes sense
print('Beta Shape :', beta.shape)
print('Atlas Shape :', atlas.shape)
#returns with beta = (22, 22, 22, 3)
#returns with atlas = (22, 22, 22)
#so we have a 22 by 22 by 22 array with 3 stimulus conditions
#and an atlas array of the same dimensions but with voxel ROI 5 different ROI not a rating of 1-5, each ROI is important

#so we are analysing 3 different stimuli, over 4 different regions, and looking to see average ROI and how similar they are to eachother
# so for example, analyze, ROI 1, with stimulus 1, as a vector of all 3 dimensions


#making a list for the cosine distances this is for the average data in each roi
#there will be 3 cosine distances 1v2 1v3 and 2v3
roi_vectors = []

#cycle through the 5 ROIs
for roi in range(1,6):
    roi_filter = (atlas == roi)
    #this will cycle through all atlas data and confirm true if equal to roi=2 for exmaple, then make a mask of true/false in the shape of each ROI based on number
    #this is stored in roi_filter
    roi_betas = beta[roi_filter]
    # this will put all the relevant beta info for EACH roi in an array nubered by voxel, and then response to stimulus

    mean_response_per_stim = np.mean(roi_betas, axis=0)
    #this gets average voxel activation per stimulus for each

    stim1 = roi_betas[:, 0] #voxel response for stim 1
    stim2 = roi_betas[:, 1] #voxel response for stim2
    stim3 = roi_betas[:, 2] # voxel response for stim 3

    #these are now separated for each stim for each relevant voxel

    #now gets the cosine distance for each condition related to eachother
    cd12 = cosine(stim1, stim2)
    cd13 = cosine(stim1, stim3)
    cd23 = cosine(stim2, stim3)

    #this makes the dissimilarity matrix so we can extract only the ones we need (top triangle)
    roi_vectors.append([cd12, cd13, cd23])
    #this needs to be a list, not 3 different variables

#now that for loop is done, we havewhat we need and we can now analyze
roi_vectors = np.array(roi_vectors) #converts to array of 5 roi, 3 stim

print(' Cosine distances vectors per ROI:' \
'', (roi_vectors))

#returns:
#Cosine distances vectors per ROI: [[0.93689252 0.82329388 0.3625235 ]
# [0.8902343  0.78586358 0.21400521]
# [0.53219521 0.33445414 0.31418117]
 #[0.4821029  0.51290044 0.61219568]
# [0.69392282 0.58130423 0.36980375]]

#output is good

#now do pearson correlation, how strong and which direction (r=strength)

from scipy.stats import pearsonr

n_roi = roi_vectors.shape[0]
correlation_matrix = np.zeros((n_roi, n_roi)) #should be 5 by 5

#loop for all the pairs of ROIs
for i in range(n_roi):
    for j in range (n_roi):
        corr, _ = pearsonr(roi_vectors[i], roi_vectors[j])
        correlation_matrix[i, j] = corr

print('nROI to ROI Pearson correlation matrix: \n ', (correlation_matrix))



#returns:
# nROI to ROI Pearson correlation matrix: 
#  [[ 1.          0.99903229  0.71435032 -0.9991741   0.98700147]
# [ 0.99903229  1.          0.68288044 -0.99642     0.97897782]
 #[ 0.71435032  0.68288044  1.         -0.74219549  0.81752865]
 #[-0.9991741  -0.99642    -0.74219549  1.         -0.99271663]
 #[ 0.98700147  0.97897782  0.81752865 -0.99271663  1.        ]]

#1.'s mean that each ROI is correlated to itself, that makes sense, nearly
#all of these are correlated strongly, but some negative


#now to analysis

roi_pairs = []
#list for the pairs and their correlation values


#loop through pairs again this time storing them
for i in range(n_roi):
    for j in range(i + 1, n_roi):  # this loops starting at 1 after i so it doesnt show the matching pairs 
        corr = correlation_matrix[i, j]
        roi_pairs.append(((i + 1, j + 1), corr))  # add 1 so that it is 1-5 like in atlas

# funtion to get abs value of correlations (BY ABS VALUE)!
def get_abso_correlation(pairandvalues):
    return abs(pairandvalues[1])
#sort
roi_pairs.sort(key=get_abso_correlation, reverse=True)

# list the pairs by order of greatest abso value
print('\nTop 5 most correlated ROI pairs: \n')

top5 = roi_pairs[:5]
for item in top5:
    pair = item[0]
    corr = item[1]

    roi1 = pair[0]
    roi2 = pair[1]

    print('ROI', roi1, 'and ROI', roi2, 'have a correlation of ', round(corr, 6))

#returns:
#Top 5 most correlated ROI pairs: 

#ROI 1 and ROI 4 have a correlation of  -0.999174
#ROI 1 and ROI 2 have a correlation of  0.999032
#ROI 2 and ROI 4 have a correlation of  -0.99642
#ROI 4 and ROI 5 have a correlation of  -0.992717
#ROI 1 and ROI 5 have a correlation of  0.987001

#good, incredibly high correlation

#now want to analyze ROI2's correlation to other ROIs witha bar graph using matplotlib

import matplotlib.pyplot as plt

roi2_corrs = correlation_matrix[1] #index1 is roi2, remember numbering in python
other_rois = [0, 2, 3, 4] #exludes matching comparison to roi2

roi_labels = ['ROI 1', 'ROI 3', 'ROI 4', 'ROI 5']
correlation_values_graph = [roi2_corrs[i] for i in other_rois]

#now can plot these

plt.bar(roi_labels, correlation_values_graph)
plt.title('Correlation of ROI 2 With Other ROIs')
plt.ylabel('Pearson r Correlation')
plt.xlabel('Other ROIs')
plt.ylim(-1.1, 1.1) # will be between -1 and 1, but want space for graph visually
plt.grid(axis='y')
plt.tight_layout()

plt.savefig('roi2_correlation_plot.png', dpi=100)
plt.show()
# % works!
# % now just interpret graph results

