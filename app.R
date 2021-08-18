#
# Simple R Shiny Dashboard App for Image recognition prediction.
#
# Purpose: Load a jpg/jepg picture and the dashboard will show what it is (or not)
#
# The app consists of 6 parts:
# A. Initial work 
# B. Load libraries
# C. Load model and data
# D. Define the User Interface
# E. Define the Data Processing
# F. Create the app


#
# Part A: Initial work. (Only needs to be done once)
# - Run the R script that creates and saves your prediction model.
# - Create a folder on your computer, and name it as you want your app the be called. (e.g. imageApp)
# - Within this folder create a folder called "www"
# - Copy the model and the label list to the www folder (folder: "animal_mod", file: "label_list.Rdata")
# - In the console type: setwd("<path_to_your_imageApp>"). This is for local use only.
# - Install the packages if needed.
#   install.packages("shiny")
#   install.packages("shinydashboard")
#   install.packages("rsconnect")


#
# Part B: Load the libraries
#

library(shiny)            # Build interactive web apps straight from R. 
library(shinydashboard)   # Use Shiny to create dashboards 
library(rsconnect)        # Deploy applications to the ShinyApps hosted service, or to RStudio Connect.
library(keras)            # A high-level neural networks API.
library(tensorflow)       # An open source software library for numerical computation using data flow graphs.
library(tidyverse)        # An opinionated collection of R packages designed for data science.


#
# Part C: Load the model and the label list
#         Also set some variables and settings
#

model <- load_model_tf("/Users/tacobakker/rstudio/imagerecognition/whatAnimal/www/animal_mod")
load("/Users/tacobakker/rstudio/imagerecognition/whatAnimal/www/label_list.Rdata")
target_size <- c(224,224,3)
options(scipen=999) #prevent scientific number formatting

#
# Part D: UI - Definition of the User Interface, with the variables.
#     The UI consists of 3 elements:
#     (1) Header
#     (2) Sidebar
#     (3) Body
#


ui <- dashboardPage(

  #(1) Header
  #    No variables, just static content
  
  dashboardHeader(title=tags$h1("What animal is this?",
                                style="font-size: 120%; font-weight: bold; color: white"),
                  titleWidth = 350),
  
  #(2) Sidebar
  #    Input variable: "input_image"    type: File
  
  dashboardSidebar(
    width=350,
    tags$p("Upload the image here."),
    fileInput("input_image","File" ,accept = c('.jpg','.jpeg')), 
    tags$br()
  ),
  
  #(3) Body
  #    Output variables:
  #      "output_image"            type: imageOutput     the uploaded photo
  #      "warning_text"            type: textOutput      show message if prediction is unclear
  #      "output_prediction_text"  type: tableOutput     show prediction output, per class
  
  dashboardBody(
    
    fluidRow(
      column(h4("Image:"),imageOutput("output_image"), width=6),
      column(h4("Result:"),tags$br(),textOutput("warning_text",), tags$br(),
             tags$p("This animal is probably a:"),tableOutput("output_prediction_text"),width=6)
    ),tags$br()
    
  ))

#
# Part E: server - Retrieve, Process and Show the data in the UI
#
# Retrieve: Input variable "input_image" is retrieved via "input$input_image"
# Process:  The input variable is loaded into the model
#           which produces a prediction (a table with percentages)
#           Note: this is a function that is called in the "Show" part.
# Show:     Output variable "output_prediction_text" is shown by "output$output_prediction_text"
#           by calling the prediction function
#           Output variable "warning_text" is shown by "output$warning_text"
#           But only if the highest prediction is lower or equal to 30%
#           Output variable "output_image" is shown by "output$output_image"
#           But it has to be retrieved first.

server <- function(input, output) {
  
  image <- reactive({image_load(input$input_image$datapath, target_size = target_size[1:2])})
  
  prediction <- reactive({
    if(is.null(input$input_image)){return(NULL)}
    x <- image_to_array(image())
    x <- array_reshape(x, c(1, dim(x)))
    x <- x/255
    pred <- model %>% predict(x)
    pred <- data.frame("Animal" = label_list, "Prediction" = t(pred))
    pred <- pred[order(pred$Prediction, decreasing=T),][1:5,]
    pred$Prediction <- sprintf("%.2f %%", 100*pred$Prediction)
    pred
  })
  
  output$output_prediction_text <- renderTable({
    prediction()
  })
  
  output$warning_text <- renderText({
    req(input$input_image)
    
    if(as.numeric(substr(prediction()[1,2],1,4)) >= 30){return(NULL)}
    warntext <- "Warning: I am not sure about this animal!"
    warntext
  })
  
  
  output$output_image <- renderImage({
    req(input$input_image)
    
    outfile <- input$input_image$datapath
    contentType <- input$input_image$type
    list(src = outfile,
         contentType=contentType,
         width = 400)
  }, deleteFile = TRUE)
  
}

#
# Part F: create the shinyApp
#

shinyApp(ui, server)