# *Image Recognition with Machine Learning in R*

## A simple guide

### By Taco Bakker

Goal: Build an Image Recognition web app, using your own photos.

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
Note that you need at least 100 jpg pictures per class,
to make the algorithm work properly.
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

## Part 4: organizing your data

`tacobakker@MacBook-Pro-van-Taco:~/Downloads/ANIMALS|⇒  ls -l`\
`total 0`\
`drwxr-xr-x@  76 tacobakker  staff   2432 Aug 18 13:40 SNAKE`\
`drwxr-xr-x@ 723 tacobakker  staff  23136 Aug 17 18:09 TIGER`\
`drwxr-xr-x@ 821 tacobakker  staff  26272 Aug 17 18:09 TORTOISE`\
`drwxr-xr-x@ 733 tacobakker  staff  23456 Aug 17 18:09 WALRUS`\
`drwxr-xr-x@ 907 tacobakker  staff  29024 Aug 17 18:09 WOLF`\
`drwxr-xr-x@ 878 tacobakker  staff  28096 Aug 17 18:09 ZEBRA`\
