
![Text](https://placehold.co/100x20?text=Text) 
![Image](https://placehold.co/100x20?text=Image) 
![Tensorflow](https://placehold.co/100x20?text=Tensorflow)
![Deep Learning](https://placehold.co/100x20?text=Deep+Learning)
![Transfer Learning](https://placehold.co/100x20?text=Transfer+Learning)
![Docker](https://placehold.co/100x20?text=Docker) 
# DS - Bootcamp - DEC22 - Rakuten Challenge
![cover](./slides/images/readme_cover.jpg) 
## Presentation
This repository contains the code for project **Rakuten** based on data issued of [Rakuten Challenge](https://challengedata.ens.fr/participants/challenges/35/) 
and developed during [Data Scientist training](https://datascientest.com/en/data-scientist-course) at [DataScientest](https://datascientest.com/).  
  

The cataloging of product listings through title and image categorization is a fundamental problem for any e-commerce marketplace. The traditional way of categorization is do it manually. However, this takes up a large portion of employees’ time and can be very expensive.

The goal of this project is to **predict product’s type code  through description and image of the product for e-commerce platform Rakuten.** 


This project was developed by the following *team* :
- Charly LAGRESLE ([GitHub](https://github.com/karolus-git/) / [LinkedIn](https://www.linkedin.com/in/charly-lagresle/))
- Olga ([GitHub](https://github.com/data-modelisation/) / [LinkedIn](https://www.linkedin.com/in/tolstolutska/))
- Mohamed BACHKAT  ([GitHub](https://github.com/mbachkat/) / [LinkedIn](https://fr.linkedin.com/in/mo-bachkat-7389451a3/))

You can browse and run the [notebooks](./notebooks). You will need to install the dependencies (in a dedicated environment) :

```
pip install -r src/requirements.txt
```

You can also see summary presentation of the project in format pdf 
[presentation.pdf](./slides/rapport.pdf) et pptx [presentation.pptx](./slides/rapport.pptx). 

## Application
The application run within a Docker container [Streamlit + FastAPI + Docker = &hearts;].

* frontend that contains  web interface created with  application Streamlit  
* backend that contains   API created with FastAPI   
* monitoring that contains application Tensorboard  

To run the docker containers :

```sh
docker-compose up --build 
```
The app should then be available at [localhost:8501](http://localhost:8501) and API documentation should be available at [localhost:8008/docs](http://localhost:8008/docs).


## Description
Rakuten Challenge contains : 

* `84 916` observations
* `27` categories to be determined 
* `0` duplicate data
* One color image per product
* Image size `500x500px` in JPG format 

The sample of the data:   

<img src="./slides/images/dataframe.svg" width="500" />

The challenge presents several *interesting research aspects* due to :
- the intrinsic noisy nature of the product labels and images 
- the typical unbalanced data distribution
- the big size of the data 
- the description of the product in different languages 

We use a supervised approach for the one-label classification problem with imbalanced distribution of labels. Therefore the metric used in this challenge to rank the model perfomance is the weighted-F1 score.

The project unfolds in next stages : 

1. Creation Text classifier 
1. Creation Image Classifier 
1. Fusion Text Classifier and Image Classifier  

Text classifier : 
* Contains text prepprocessing and text vectorization using Natural Language Processing (NLP).   
* Base on the *Neural_Embedder* text model. 

Image classifier : 
* Base on *CNN* image model
* Use transfer learning with the *MobileNetV2* model loaded with pre-trained weights on *ImageNet*. 

Fusion model has next architecture :  

<img src="./notebooks/images/fusion_methodology.png" width="300" />

The fusion model uses the image model to categorize products where the text model underperformed. The global **weighted-F1 score is 82.2%** and **all categories exceed the 55% score**. 


