#
# Example Image Recognition with Machine Learning
# https://www.kaggle.com/biancaferreira/african-wildlife
# https://www.kaggle.com/jerrinbright/cheetahtigerwolf 
#

# 
# The following must be typed into the console:
# install_tensorflow(extra_packages="pillow")
# install_keras()

library(tidyverse)
library(keras)
library(tensorflow)
library(reticulate)

# Download the African Wildlife dataset from Kaggle. 
# We limit our engine to 4-5 species
# So you should have folders “train” and “test” with 4 subfolders 
# each with images for training and testing for each of the 4 species. 
# (There is also a validation folder 
# but we will create the training/validation split in the code to make it more generalizable).

setwd("~/machinelearning/data/testwild")          # folder where the images are stored
label_list <- dir("~/machinelearning/data/testwild/train/") # a list of all species in the correct order 
output_n <- length(label_list)                    # total number of species
save(label_list, file="label_list.Rdata")         # save it in a file for later
path_train <- "~/machinelearning/data/testwild/train/"
path_test <- "~/machinelearning/data/testwild/test/"

width <- 224                                 # resizing the images to 224 pixels
height<- 224                                 # width and height
target_size <- c(width, height)
rgb <- 3 #color channels

test_image1 <- image_load("~/Pictures/animal.jpg", target_size = target_size)

# After we set the path to the training data, 
# we use the image_data_generator() function to define the preprocessing of the data.


train_data_gen <- image_data_generator(rescale = 1/255, 
                                       validation_split = .2)   # reserve 20% of the data for a validation dataset

# The flow_images_from_directory() function batch-processes the images 
# with the above defined generator function. 
# With the following call you assign the folder names in your “train” folder as class labels, 
# which is why you need to make sure that the sub-folders are named according to the bird species 
# as shown in the second picture above. 
# We create two objects for the training and validation data.
# It should give us the confirmation about how many images were loaded:

train_images <- flow_images_from_directory(path_train,
                                           train_data_gen,
                                           subset = 'training',
                                           target_size = target_size,
                                           class_mode = "categorical",
                                           shuffle=F,
                                           classes = label_list,
                                           seed = 2021)

validation_images <- flow_images_from_directory(path_train,
                                                train_data_gen, 
                                                subset = 'validation',
                                                target_size = target_size,
                                                class_mode = "categorical",
                                                classes = label_list,
                                                seed = 2021)

# Note again that we do not use separate folders for training and validation in this example 
# but rather let keras reserve a validation dataset via a random split.

table(train_images$classes)              # check if the set looks ok

# pip3 install --upgrade pip
# pip3 install --user numpy scipy matplotlib ipython jupyter pandas sympy nose


plot(as.raster(train_images[[3]][[1]][25,,,]))   # show a random example of a picture

# We are going to train a convoluted neural network (CNN).
# Now, the great flexibility of neural networks that enables them to learn any kind of function comes at a cost: 
# There are millions of different ways to set up such a model, 
# and depending on the values of parameters that most people have no idea what they are doing, 
# your model might end up with anything between 3% and 99% accuracy for the task at hand.
# Luckily, there is a way to quickly generate good baseline results: Loading pre-trained models 
# that have proven well in large-scale competitions such as ImageNet.
# Here, we load the xception-network with the weights pre-trained on the ImageNet dataset – 
# except for the final layer 
# (which classifies the images in the ImageNet dataset) which we’ll train on our own dataset. 
# include_top” is set to FALSE, to make sure we train the last layer ourselves.

mod_base <- application_xception(weights = 'imagenet', 
                                 include_top = FALSE, input_shape = c(width, height, 3))
freeze_weights(mod_base) 

# Now let’s write a small function that builds a layer on top of the pre-trained network 
# and sets a few parameters to variabes that we can later use to tune the model:

model_function <- function(learning_rate = 0.001, 
                           dropoutrate=0.2, n_dense=1024){
  
  k_clear_session()
  
  model <- keras_model_sequential() %>%
    mod_base %>% 
    layer_global_average_pooling_2d() %>% 
    layer_dense(units = n_dense) %>%
    layer_activation("relu") %>%
    layer_dropout(dropoutrate) %>%
    layer_dense(units=output_n, activation="softmax")
  
  model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_adam(lr = learning_rate),
    metrics = "accuracy"
  )
  
  return(model)
  
}

# We let this model compile once with our base values and inspect its architecture:

model <- model_function()
model

# let’s start with training the model as is:
# Depending on your hardware this may take a few minutes up to an hour.

batch_size <- 32
epochs <- 6

hist <- model %>% fit_generator(
  train_images,
  steps_per_epoch = train_images$n %/% batch_size, 
  epochs = epochs, 
  validation_data = validation_images,
  validation_steps = validation_images$n %/% batch_size,
  verbose = 2
)

model %>% save_model_tf("animal_mod")

# Let’s see how well our model classifies the species in the hold-out test dataset. 
# We use the same logic as above, creating an object with all the test images scaled to 224×224 pixels 
# as set in the beginning. 
# Then we evaluate our model on these test images:



test_data_gen <- image_data_generator(rescale = 1/255)

test_images <- flow_images_from_directory(path_test,
                                          test_data_gen,
                                          target_size = target_size,
                                          class_mode = "categorical",
                                          classes = label_list,
                                          shuffle = F,
                                          seed = 2021)

# We can also upload a custom image to see what our model predicts. 
# (making sure it’s not one of the training images) and fed it to the model:

model %>% evaluate_generator(test_images, 
                             steps = test_images$n)




x <- image_to_array(test_image1)
x <- array_reshape(x, c(1, dim(x)))
x <- x/255
pred <- model %>% predict(x)
pred <- data.frame("Species" = label_list, "Probability" = t(pred))
pred <- pred[order(pred$Probability, decreasing=T),][1:5,]
pred$Probability <- paste(format(100*pred$Probability,2),"%")
pred

