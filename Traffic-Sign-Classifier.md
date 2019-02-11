# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)
**sample images from web are stored in images_from_web folder
**visualisation images are stored in examples.
[//]: # (Image References)

[image1]: ./examples/data_viualisation.png "Visualization"
[image2]: ./images_from_web/roundabout.png "Traffic Sign 1"
[image3]: ./images_from_web/keep_right.png "Traffic Sign 2"
[image4]: ./images_from_web/no_passing.png "Traffic Sign 3"
[image5]: ./images_from_web/priority_road.png "Traffic Sign 4"
[image6]: ./images_from_web/dangerous_curve_right.png "Traffic Sign 5"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! 

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is (34799, 32, 32, 3)
* The size of the validation set is (4410, 32, 32, 3)
* The size of test set is (12630, 32, 32, 3)
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 
[image1]: ./examples/data_visualisation.jpg "Visualization"
It is a bar chart showing how the data 

**image included in notebook.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As part of the preprocessing step, I combined all the dataset and reshuffled to generate the training, validation and testing datasets using the train_test_split function. Follwed by normalizing the data so the data has close to zero mean and equal variance. In future, I would like to experiment with data augmentation to generate additional data. As seen in the above histogram in the notebook, the distribution of data between various labels is uneven. 
As explained in the tensorflow lesson, neural networks perform much better if the inputs are normalized. I performed this normalization by substracting pixel_depth from input and dividing by pixel depth. Post normalization, the input values are b/w -1 and 1 In addition to normalizing, I also shuffled X_train and y_train since there are 43 classes in the dataset and we want the data to be randomly distributed


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 5x5 	    | 1x1 stride, same padding, outputs 10x10x16.	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs  5x5x16 					|
| Flatten               | outputs 400  									|               |  
| Fully connected		| outputs 120  									|
| RELU					|												|
| Fully connected		| outputs 84  									|
| RELU					|												|
| Fully connected		| outputs 43  									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an adam optimizer, with a batch size of 256, learning rate of 0.002, running it for 30 epochs .

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 100%
* validation set accuracy 95.4%
* test set accuracy of 98.6%

**If an iterative approach was chosen:

I followed an iterative approach expecting an accuracy > 0.93. Since I already had an implementation for the LeNet-5 architecture explained in lessons, I just went ahead with that in the beginning by just modifying the image channel depth to 3. Later, more fully connected layers were added to the bottom **but could not acheive more than 90% accuracy.What helped me in this situation was combining all the dataset to create a new training dataset shuffling and normalizing of data in the pre-processing stages.this helped me go beyond 95% in training accuracy.

The Lenet architecture starts with a convolution layer with patch size of 5x5, stride of 1 and a depth of 6. The convolution layer is multiplied by a RELU activation. After the convolution layer is a max pool layer with kernel size of 2 and stride of 2. The model has another set of convolution + max pool layer with similar patch size, stride and kernel size but with a depth of 16. This layer also uses Relu activation. Finally the model has 3 fully connected layers including the output layer. The three fully connected layers have 400, 120 and 84 neurons respectively and use Relu activation b/w them. The output layer returns an output of length 43 corresponding to the possible classes in labels

The learning rate hyperparameter was increased from 0.001 to 0.002 for faster convergence since I observed that with the default learning rate, it required more number of epochs to reach the optimum accuracy.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

**images from the web are stored in images_from_web folder
*images are resized images (32x32x3) 


[image7]: ./images_from_web/roundabout.png "Traffic Sign 1"
This image might be difficult to classify because of its low contrast ratio.
[image8]: ./images_from_web/keep_right.png "Traffic Sign 2"
[image9]: ./images_from_web/no_passing.png "Traffic Sign 3"
[image10]: ./images_from_web/priority_road.png "Traffic Sign 4"
[image11]: ./images_from_web/dangerous_curve_right.png "Traffic Sign 5"

in general The original images from the German dataset are not very good quality. They are blurred and it can be difficult to make out the actual sign.



#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

|        Image                      |     Prediction                                | 
|:---------------------------------:|:---------------------------------------------:| 
| Roundabout mandatory              | Roundabout mandatory                          | 
| Keep right                        | Keep right                                    |
| No passing                        | No passing                                    |
| Priority road                     | Priority road                                 |
| Roundabout mandatory              | Roundabout mandatory                          |


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 98.6%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, the model is relatively sure that this is Roundabout mandatory (probability of 0.999999881), and the image does contain a round about sign. The top five soft max probabilities were

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| 0.999999881           | 40,Roundabout mandatory                       | 
|  0                    | 35,Ahead only                                 |
|  0                    | 33,Turn right ahead                           |
|  0                    | 34,Turn left ahead                            |
|  0                    | 12,Priority road                              |

a similar probabilities were observed for rest of the images






### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


