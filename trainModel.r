#
# Example Image Recognition with Machine Learning
#


library(tidyverse)
library(keras)
library(tensorflow)
library(reticulate)

#
# Put your own values below!!!
#

path_data <- "~/machinelearning/data/testwild/"   # The folder where all your data is stored
name_of_model <- "animal_mod"                     # The name of your model
name_of_class_list <- "label_list.Rdata"          # The name of your class list

#
# Define the variables
# No need to change anything unless you know what you're doing
#

path_train <- paste(path_data,"train/",sep="")
path_test <- paste(path_data,"test/",sep="")
setwd(path_data) 
label_list <- dir(path_train)                     # a list of all classes 
output_n <- length(label_list)                    # total number of classes
save(label_list, file=name_of_class_list)         # save it in a file for later
width <- 224                                      # resizing the images to 224 pixels
height<- 224                                      # width and height
target_size <- c(width, height)
rgb <- 3  

#
# Data preparation
#

# set the variables for processing in the data generator
train_data_gen <- image_data_generator(rescale = 1/255, validation_split = .2)   

# Generate array of images for training
train_images <- flow_images_from_directory(path_train,
                                           train_data_gen,
                                           subset = 'training',
                                           target_size = target_size,
                                           class_mode = "categorical",
                                           shuffle=F,
                                           classes = label_list,
                                           seed = 2021)
# Generate array of images for validation
validation_images <- flow_images_from_directory(path_train,
                                                train_data_gen, 
                                                subset = 'validation',
                                                target_size = target_size,
                                                class_mode = "categorical",
                                                classes = label_list,
                                                seed = 2021)

# Generate a model based on a standard learning model (imgagenet) for image recognition
# We do not include the top layer. That will be our own images.
# Freeze the weigths we just calculated so our top layer doesn't mess with them.
mod_base <- application_xception(weights = 'imagenet', 
                                 include_top = FALSE, input_shape = c(width, height, 3))
freeze_weights(mod_base) 

# Function to build an empty layer on top of the pre-trained network
#
# k_clear_session - avoid clutter from old models / layers.
# keras_model_sequential - function to add layers to a model
# layer_global_average_pooling_2d - scale down the complexity of the layer (tensor) to 2D
# layer_dense - another function to unify the layer (tensor)
# A tensor is a container which can house data in N dimensions, along with its linear operations
# layer_activation - in a neural network, the activation function is responsible for transforming 
#   the summed weighted input from the node into the activation of the node or output for that input.
# relu - The rectified linear activation function or ReLU for short is a piecewise linear function that will 
#   output the input directly if it is positive, otherwise, it will output zero.
# layer_dropout - value between 0 and 1. Fraction of the input units to drop to avoid overfitting.
# Overfitting - "the production of an analysis that corresponds too closely or exactly to a particular set of data, 
#  and may therefore fail to fit additional data or predict future observations reliably".
# softmax - normalizes into a probability distribution.
#
# Compile defines the loss function, the optimizer and the metrics. 
# The purpose of loss functions is to compute the quantity that a model should seek to minimize during training.
# optimizer - includes the method to train your machine/deep learning model.
# metrics - A metric is a function that is used to judge the performance of your model.

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

# Here we create the model by calling the function
# fit - the function that does the actual training of the model
# epochs - number of max iterations on the input data
# The model is not trained for a number of iterations given by epochs, 
# but merely until the epoch of index epochs is reached.

model <- model_function()
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
model %>% save_model_tf(name_of_model)

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

