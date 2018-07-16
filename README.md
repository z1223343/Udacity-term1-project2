# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./image_1.jpg
[image2]: ./image_2.jpg
[image3]: ./image_3.jpg
[image4]: ./image_4.jpg
[image5]: ./image_5.jpg
[image6]: ./plot.jpg
[image7]: ./plot2.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/z1223343/udacity-term1-project2/blob/master/Traffic_Sign_Classifier.ipynb) 

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is （32,32,3）
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. These are the bar charts showing the probability of each sign in the training, validation and testing data set respectively. 

![][image6]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. 

I normalized all the input pictures to make the value between -1 and 1, because theoretically normaliztion would make it easier for the neural network system to find the correct direction when applying gradient decline algorithm.
I kept all the 3 channels of input RGB images, because I think the color maybe helpful to do recognition.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, VALID padding, outputs 10x10x16     		|
| RELU  |                      |
|Max pooling      |    2x2 stride, outputs 5x5x16           |
|Flatten   |      outputs 400|
| Fully connected		| outputs 120        									|
| RELU			|         									|
|	Dropout					|	Keep rate: 0.5											|
|		Fully connected				|			outputs 84									|
|RELU  |     |
|Dropout   |         Keep rate: 0.5     |
|Fully connected  | outputs 43    |
 


#### 3. Describe how you trained your model. 

To train the model, I used:
* Constant learning rate = 0.0008
* Number of epochs = 30
* Batch size = 128
* Optimizer: AdamOptimizer
* Softmax and one-hot function to process logits

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93.

My final model results were:

* training set accuracy of 0.996
* validation set accuracy of 0.957
* test set accuracy of 0.938

I chose LeNet CNN model as the architecture.
* In the LeNet Lab project, it works pretty well with the recognition of 10 digits using MNIST data set, so it should be applicable to traffic sign recognition in this project.
* Based on the architecture of LeNet, I tweaked the learning rate with the epoches I used, I also find the value of 'sigma' in the function **truncated_normal()**. The accuraccy is below 0.1 when I didn't set initial sigma value or make sigma as big as '0.9', and the accuraccy is also not good when 'sigma' is too low. Results are good when 'sigma' is near 0.1, and in this project I set it as 0.11.
* I also tried to improve it by using dropout algorithm, and it works well to prevent the CNN overfitting. Finally I set the keep rate as 0.5.

![][image7]

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![][image1] ![][image2] ![][image3] 
![][image4] ![][image5]


The first and second images might be difficult to classify because there are speed limit traffic signs for 20/30/50/70/90/120, and the number in the pictures are very similar for CNN to recognize.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| speed limit: 20km/h      		| speed limit: 30km/h   									| 
| speed limit: 70km/h     			| speed limit: 70km/h 										|
| Stop					| Stop											|
| No entry	      		| No entry					 				|
| round mandatory			| round mandatory      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. It is lower than the accuracy of the test set as 93.7%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction.

For the first image "speed limit 20 km/h"

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .61         			| Speed limit (30km/h)   									| 
| .32     				| Speed limit (20km/h) 										|
| .05					| Speed limit (120km/h)											|
| .00      			| End of all speed and passing limits					 				|
| .00				    | Children crossing     							|

For the second image "speed limit 70 km/h"

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .89         			| Speed limit (70km/h)  									| 
| .10     				| 	Speed limit (20km/h) 										|
| .00					| General caution											|
| .00      			| Bicycles crossing					 				|
| .00				    |Road narrows on the right     							|

For the third image "Stop"

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Stop  									| 
| .00     				| Bumpy road 										|
| .00					| No entry											|
| .00      			| No passing					 				|
| .00				    |Road work     							|

For the forth image "No entry"

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| No entry 									| 
| .00     				| Stop 										|
| .00					| Bicycles crossing											|
| .00      			| Bicycles crossing					 				|
| .00				    |Road narrows on the right     							|

For the fifth image "Roundabout mandatory"

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Roundabout mandatory  									| 
| .00     				| Keep right 										|
| .00					| Go straight or left											|
| .00      			| Go straight or right					 				|
| .00				    |Wild animals crossing     							|
