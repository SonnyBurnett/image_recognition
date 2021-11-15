#
# Example Image Recognition with Machine Learning
#


library(tidyverse)
library(keras)
library(tensorflow)
library(reticulate)

# set warn to 0 to see the warnings again
options(warn=-1)
options(digits = 2)
getOption("digits")
#
# Put your own values below!!!
#

path_data <- "~/machinelearning/data/testwild/"   # The folder where all your data is stored
name_of_model <- "animal_mod"                     # The name of your model
name_of_class_list <- "label_list.Rdata"          # The name of your class list

#
# 1. Define the variables
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
# 2. Data preparation
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

# 3. Create a model

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

model_function <- function(){
  k_clear_session()
  
  model <- keras_model_sequential() %>%
    mod_base %>% 
    layer_global_average_pooling_2d() %>% 
    layer_dense(units = 1024) %>%
    layer_activation("relu") %>%
    layer_dropout(0.2) %>%
    layer_dense(units=output_n, activation="softmax")
  
  # Compile the model
  model %>% compile(
    loss = 'categorical_crossentropy',
    optimizer = 'adam',
    metrics = 'accuracy'
  )
  
  return(model)
}

# Here we create the model by calling the function

model <- model_function()

# 4. Train the model

# fit - the function that does the actual training of the model
# epochs - number of max iterations on the input data
# The model is not trained for a number of iterations given by epochs, 
# but merely until the epoch of index epochs is reached.

batch_size <- 32
epochs <- 6

#hist <- model %>% fit_generator(
  
hist <- model %>% fit(
  train_images,
  steps_per_epoch = train_images$n %/% batch_size, 
  epochs = epochs, 
  validation_data = validation_images,
  validation_steps = validation_images$n %/% batch_size,
  verbose = 6
)
model %>% save_model_tf(name_of_model, overwrite = TRUE,
                        include_optimizer = TRUE,
                        signatures = NULL,
                        options = NULL)

# 5. Test the model

# Let’s see how well our model classifies the test dataset. 
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

# get the names of the folders. This is also the name of the class to predict
test_names <- list.files(path = path_test, pattern = NULL, all.files = FALSE,
                         full.names = FALSE, recursive = TRUE,
                         ignore.case = FALSE, include.dirs = FALSE, no.. = FALSE)
tn <- sapply(strsplit(test_names,"/"), `[`, 1)

# predict all test images
pred <- model %>% predict(test_images)

# make a nice matrix to see how well the model does it
# the first column is the class we should predict
# the other columns show the prediction certainty per class
pred2 <- round(pred, digits = 1)
colnames(pred2) <- label_list
rownames(pred2) <- tn
pred2
table(test_images$classes)
plot(as.raster(test_images[[1]][[1]][1,,,]))
typeof(test_images[1])
test_images[1][1]
