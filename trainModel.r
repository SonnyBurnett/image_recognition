#
# Example Image Recognition with Machine Learning
#


library(tidyverse)
library(keras)
library(tensorflow)
library(reticulate)


path_data <- "~/machinelearning/data/testwild/"   # The folder where all your data is stored
name_of_model <- "animal_mod"                     # The name of your model
name_of_class_list <- "label_list.Rdata"          # The name of your class list

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
test_image1 <- image_load(path_test_image, target_size = target_size)
train_data_gen <- image_data_generator(rescale = 1/255, validation_split = .2)   

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

mod_base <- application_xception(weights = 'imagenet', 
                                 include_top = FALSE, input_shape = c(width, height, 3))
freeze_weights(mod_base) 

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

