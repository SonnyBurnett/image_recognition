# *Image Recognition with Machine Learning in R*

## A simple guide

### By Taco Bakker

Goal: Build an Image Recognition web app, using your own selected photos.

---

## Part 1: prerequisites

1. A good performing computer (Windows, Mac, Linux)
2. Internet connection
3. Basic knowledge of IT, programming and machine learning would be nice
4. A good learning mood

## Part 2: prepare your computer

*Install R*

[Download R Mac](https://cran.r-project.org/bin/macosx/)

[Download R Windows](https://cran.r-project.org/bin/windows/base/)

*Install R Studio*

[Download Rstudio](https://www.rstudio.com/products/rstudio/download/)

For this workshop you need some dependencies from *Python*.
If you don't have Python on your machine install it:

[download python mac](https://www.python.org/downloads/macos/)

[download python windows](https://www.python.org/downloads/windows/)

## Part 3: find a project

What is it that you want to recognise?
What tree is this? What animal? What city?
Note that you need at least 100 jpg pictures per class to make the algorithm work properly.
And preferably more. (say 500).
Downloading these one-by-one will take you a lot of time.
So try to find a ready-to-use dataset.
Kaggle is my favorite, but there are many others.

[Kaggle](https://www.kaggle.com/),
[OpenImages](https://storage.googleapis.com/openimages/web/index.html),
[CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html),
[ImageNet](https://image-net.org/),
[the Visual Genome](http://visualgenome.org/),
[COCO](https://cocodataset.org/#home)

In the example I have used these datasets:
- https://www.kaggle.com/biancaferreira/african-wildlife
- https://www.kaggle.com/jerrinbright/cheetahtigerwolf

## Part 4: organizing your data

Your data must be organized in separate folders per class.
In the example below I have selected 6 animals.
Note: limit your classes to 5-6 when you start.
Deep Learning requires a lot of computer power.
If your computer can handle it, you can add more classes.

Every folder contains a number of jpg photos of an animal species.
Note that the SNAKE has a very low amount of files (76).

`tacobakker@MacBook-Pro-van-Taco:~/Downloads/ANIMALS|⇒  ls -l`\
`total 0`\
`drwxr-xr-x@   76  tacobakker  staff   2432 Aug 18 13:40 SNAKE`\
`drwxr-xr-x@ 723 tacobakker  staff  23136 Aug 17 18:09 TIGER`\
`drwxr-xr-x@ 821 tacobakker  staff  26272 Aug 17 18:09 TORTOISE`\
`drwxr-xr-x@ 733 tacobakker  staff  23456 Aug 17 18:09 WALRUS`\
`drwxr-xr-x@ 907 tacobakker  staff  29024 Aug 17 18:09 WOLF`\
`drwxr-xr-x@ 878 tacobakker  staff  28096 Aug 17 18:09 ZEBRA`

We now need to split the data in a TRAIN set and a TEST set.
- Create a train folder and a test folder.
- Copy all downloaded folders to the train folder.
- In the test folder, create a sub-folder for every class.
- Move at least 5 pictures per class from the train folder to the test folder.

You can use the script *splitdata.sh* to do all of this automatically.

Suppose you store the data in a folder called ANIMALS,
the structure should look something like this:

`ANIMALS`\
`├── test`\
`│   ├── SNAKE`\
`│   ├── TIGER`\
`│   ├── TORTOISE`\
`│   ├── WALRUS`\
`│   ├── WOLF`\
`│   └── ZEBRA`\
`└── train`\
`        ├── SNAKE`\
`        ├── TIGER`\
`        ├── TORTOISE`\
`        ├── WALRUS`\
`        ├── WOLF`\
`        └── ZEBRA`

## Part 5: Prepare Rstudio

- [Copy the example code from GitHub](https://github.com/SonnyBurnett/image_recognition)
- Open Rstudio
- Open the file trainModel.R
- Save it under your own name in a folder of your choice

Now you need to install some R packages.
- Go to the "Console" in Rstudio. (Usually the lower left window)
- Note: This window will have 3 tabs. Console, Terminal, Jobs.
- type: (don't get scared by red text)
- `install_tensorflow(extra_packages="pillow")`
- `install_keras()`
- `install.packages("tidyverse")`
- `install.packages("reticulate")`

- Now go to tab "Terminal". Here we will install some Python dependencies.
- type: (either pip3 or pip)
- `pip3 install --upgrade pip`
- `pip3 install --user numpy scipy matplotlib ipython jupyter pandas sympy nose`

*TensorFlow* is an end-to-end open source platform for machine learning.
*Keras* is a deep learning API written in *Python*,
running on top of the machine learning platform *TensorFlow*.
Keras needs some Python libraries,
which we install with pip.

The *reticulate* package provides tools to use python in R.
*Tidyverse* is a collection of R packages designed for data science.

## Part 6: Prepare the R script that generates the model
