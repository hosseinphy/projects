---
layout: single
title: "Image Classification"
date: 2021-11-19 12:00:00 -0000
categories: Neural-Network, CNN, Transfer-Learning Image-classification TensorFlow Keras  
excerpt: Classifying images using series of neural networks. 
---

## Summary
In this project, we classify series of images into ten classes using multi-layer perceptron, convolutional neural network, and by using transfer learning from a pre-trained model (Inception model).


## Data format
We used `CIFAR-10` data set that consists of 60,000 images, each 32x32 color pixels, each belonging to one of ten classes. We used 50,000 images for training and 10,000 for validation of our models. 

<br>
 
 <div align="center">
  <img src="/assets/images/blogs/ten_classes.png" width="600px" height="240" alt="Photo of a lighthouse.">
  <p>Ten sample images representing ten classes</p>
 </div>

<br>

## Transfer learning 
In this approach we computed the latent vectors (the result of the images run through a pre-trained network) ahead of time and use those as our input features to the last few dense layers. The pre-trained network that we used here is [`Inception`](https://keras.io/applications/). 

To build a classification model using keras API the following steps were taken:
1. Load the `inception` model:
  The following code snippet load the model and omit its classification layer, which is not required for our purposes. 
  ```python
  inception = tf.keras.applications.inception_v3.InceptionV3(include_top=True, input_shape=(299, 299, 3))
  inception = tf.keras.Model([inception.input], [inception.layers[-2].output]) # manually discard prediction layer
  ```
2. Upscale our images from 32x32 to `inception` native image shape: 299x299, using `tf.image.resize`

3. Feed the resized image layer to inception model and save the calculated latent vectores to disk. Since we are not interested in re-training the inception mode, we freez all the layers of the model.


```python
 for layer in inception.layers:
    layer.trainable = False  
``` 
The summary of latent vector calculation model is shown in the snippet below:

```markdown
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_3 (InputLayer)         [(None, 32, 32, 3)]       0         
_________________________________________________________________
resizing_1 (Resizing)        (None, 299, 299, 3)       0         
_________________________________________________________________
functional_1 (Functional)    (None, 2048)              21802784  
=================================================================
Total params: 21,802,784
Trainable params: 0
Non-trainable params: 21,802,784
```

4. Finally, we load features as an input to our model with a classification layer to make predictions

```markdown
Epoch 1/10
834/834 [==============================] - 1s 2ms/step - loss: 0.5941 - accuracy: 0.7976
Epoch 2/10
834/834 [==============================] - 1s 2ms/step - loss: 0.4592 - accuracy: 0.8406
Epoch 3/10
834/834 [==============================] - 2s 2ms/step - loss: 0.4161 - accuracy: 0.8552
Epoch 4/10
834/834 [==============================] - 2s 2ms/step - loss: 0.3979 - accuracy: 0.8605
Epoch 5/10
834/834 [==============================] - 1s 2ms/step - loss: 0.3686 - accuracy: 0.8704
Epoch 6/10
834/834 [==============================] - 1s 2ms/step - loss: 0.3533 - accuracy: 0.8754
Epoch 7/10
834/834 [==============================] - 2s 2ms/step - loss: 0.3348 - accuracy: 0.8822
Epoch 8/10
834/834 [==============================] - 2s 2ms/step - loss: 0.3225 - accuracy: 0.8843
Epoch 9/10
834/834 [==============================] - 2s 2ms/step - loss: 0.3077 - accuracy: 0.8901
Epoch 10/10
834/834 [==============================] - 2s 2ms/step - loss: 0.2949 - accuracy: 0.8935
```
 
The picture below shows the results of our classification model predictions for few images:

<div align="center">
  <img src="/assets/images/blogs/pred_labels.png" width="500px" height="200">
</div>
