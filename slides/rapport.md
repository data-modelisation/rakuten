---
marp: true

class: 
    #- lead
    #- invert
footer:  DST - Bootcamp - DEC22 - Rakuten Challenge
header: ''
paginate: true

---
<style>
@import 'default';
section {
  background-image: none;
  font-family:  'Verdana'; 
  font-weight: normal; 
  font-size: 1.5em;
  padding-top: 90px;
  padding-left: 40px;
  padding-right: 40px;
   
}
/* https://github.com/marp-team/marpit/issues/271 */
section::after {
  #font-weight: 700;
  font-size: 0.7em;
  content: attr(data-marpit-pagination) '/' attr(data-marpit-pagination-total);
}
section.lead h1, section.lead h2, section.lead h3{
  color: #800000;
  text-align: left;
}
header {
  background-color: #800000;
  color: #fff;
  font-size: 1em;
  font-weight: 700;
  padding: 0.2em 2em 0.2em 2em;
}


blockquote {
  max-width: 90%;
  border-top: 0.1em dashed #555;
  font-size: 60%;
  position: absolute;
  bottom: 20px;
}
blockquote::before {
  content: "";
}
blockquote::after {
  content: "";
}
</style>    
<!--
_class: 
    - lead
_paginate: false  
_footer: ''
_header: '' 
-->


![width:600px center](https://challengedata.ens.fr/logo/public/RIT_logo_big_YnFAcFo.jpg) __Participants :__
Olga TOLSTOLUTSKA
Mohamed BACHKAT
Charly LAGRESLE
![bg left height: 100px](https://img.freepik.com/free-vector/ai-technology-brain-background-vector-digital-transformation-concept_53876-112224.jpg?w=826&t=st=1678478673~exp=1678479273~hmac=30056e96f26cfed14acb6d22fe55d7329c23fe2998a8ee425cc206b63a812474)

![height:60px right](https://i0.wp.com/datascientest.com/wp-content/uploads/2022/03/logo-2021.png?w=429&ssl=1)

__Mentor :__ Manu POTREL
__Promotion:__ DST Bootcamp DEC22

___

<!--
_header: 'Context' 
-->

![bg height:80%](https://rit.rakuten.com/wp-content/uploads/2022/03/RakutenDataChallenge_RIT_Paris-1024x493.jpg)



___

<!--
_header: 'Description des données' 
-->

* 27 variables cibles 
* 84 916 observations: `des données textuelles ainsi que des images`.
* Pas de duplications des données 
* Les données textuelles sont divisés en deux colonnes : `designation`   et `description`. Elles represent un titre du produit et sa decription. 
* Le titre du produit est composé de 4 à 54 mots
* La  description est plus longs et contient entre 0 (certaines descriptions sont vides) et 2 068 mots
* Images : couleur, `500x500px` encodées au format JPG 

---
<!--
_header: 'Description des données / Nombre de produits par catégorie' 
-->
![bg height:90%](../notebooks/images/images_category.png)

___

<!--
_header: 'Description des données /Les catégories et leurs descriptions' 
-->

|Catégorie | Description|Catégorie| Description|Catégorie| Description|
|---:|------------|----:|------------|-------------------:|------------|
| 10 | Livre d'occasion   | 1301 | Chaussette | 2462 | Jeu oldschool |
| 40 | Jeu Console        | 1302 | Gadget     | 2522 | Bureautique |
| 50 | Accessoire Console | 1320 | Bébé       | 2582 | Décoration |
| 60 | Tech               | 1560 | Salon      | 2583 | Aquatique |
| 1140 | Figurine         | 1920 | Chambre    |2585 | Soin et Bricolage |
| 1160 | Carte Collection | 1940 | Cuisine    | 2705 | Livre neuf |
| 1180 | Jeu Plateau      | 2060 | Chambre enfant | 2905 | Jeu PC |
| 1280 | Déguisement      | 2220 | Animaux    | | |
| 1281 | Boite de jeu     | 2280 | Affiche    | | |
| 1300 | Jouet Tech       | 2403 | Revue    | | |

___
<!--
_header: 'Exploration des donnéess / Target' 
-->

![bg width:90%](../notebooks/images/imbalanced.png)

___
<!--
_header: 'Exploration des donnéess / Text' 
-->
![bg width:100%](../notebooks/images/words.png)

___
<!--
_header: 'Exploration des donnéess / Text' 
-->
![bg width:100%](../notebooks/images/lang.png)
![bg width:95%](../notebooks/images/common_words.png)
___

<!--
_header: 'Exploration des donnéess / Images' 
-->
![bg width:90%](../notebooks/images/white.png)
![bg width:90%](../notebooks/images/mask.png)
___
<!--
_header: 'Préparation des données / Text' 
-->
<style scoped>
table {
  font-size: 17px;
}
section {
  font-size: 17px;
}
</style>
L'exemple de transformations appliquées : 
* `designation` : Une table très jolie! 
* `description` : <ul><li>\&#43;Dimensions : 60 x 33 cm</li>

| Etape                                                 |     Résultat                                   | 
| :----- | :----------------------------------------------- | 
| Fusion de deux colonnes                               | Une table très jolie! <ul><li>\&#43;Dimensions : 60 x 33 cm</li> | 
| Détection la langue  et traduction en français        | Une table très jolie! <ul><li>\&#43;Dimensions : 60 x 33 cm</li> | 
| Suppression les balises html                          | Une table très jolie! Dimensions : 60 x 33 cm  | 
| Suppression des caractères non alphanumériques          | Une table très jolie Dimensions x cm           |
| Passage en minuscule                                  | une table très jolie dimensions x cm           |
| Encodage                                              | une table tres jolie dimensions x cm           |
| Les mots d'un caractère                               | une table tres jolie dimensions cm             |
| Suppression des *stopwords*                           | table tres jolie dimensions cm                 | 
| Extraction de la racine des mots                      | tabl tres jol dimens cm                        | 
| Vectorisation du texte via un `Tokenizer`             | [6, 1, 2, 4, 5 ]                               | 
___

<!--
_header: 'Préparation des données / Images' 
-->
__ImageDataGenerator__:
* streaming per batch : les images sont transmises sous de batchs ce qui évite de traiter l'ensemble des données d'un coup
* augmentation de données via les transformation appliqués 
* rédimensionnement en taille 224x224
* application de la fonctionne `preprocess_input` spécifique pour chaque modèle 

![bg right width:80%](../notebooks/images/rescale.png)
___
<!--
_header: 'Les modèles / Deep learning / Text ' 
-->
![bg height:95%](images/models_dl_text.jpg)

___
<!--
_header: 'Les modèles / Deep learning / Fusion ' 
-->

![width:90%](../notebooks/images/fusion_methodology.png)
Explication de fusion ....
___
<!--
_header: 'Les modèles / Deep learning / Fusion ' 
-->
![bg width:42%](images/models_dl_text.jpg)
![bg]()
![bg width:32%](images/models_fusion.jpg)
___
<!--
_header: 'Analyse du meilleur modèle' 
-->
Pas d'impacte sur les performances réduites du modèle d'image.
  * Toutes les catégories dépassent le score de 54% et 
  * Une catégorie sur trois dépasse le score de 90%

Le modèle concaténé s'aide du modèle d'image pour catégoriser les produits où le modèle de texte sous-performait : 
  * La catégorie 1080 (Jeu Plateau) gagne 25 points
  * La catégorie 2705 (Livre neuf) gagne 23 points

![bg right:58% width:98%](images/models_fusion_crosstab.jpg)
___
<!--
_header: 'Limites' 
-->
Le projet est un projet mêlant de l'analyse de texte et du traitement d'images : des notions poussées de deep-learning sont nécessaires à la compréhension et l'implémentation de telles techniques.
De nombreux limites sont apparus tout au long de ce projet :

* L'accès à des ressources de calcul de type GPU ou TPU nous a été quasi impossible, notamment via Google Collab. 
* L'accès aux 84 916 images, stockées dans un Google Drive et nécessaires à l'entraînement du modèle d'images, était érattique : de nombreuses coupures de ce lien entre Google Drive et Google Collab ont entraîné ici aussi une grande perte de temps et une grande frustration.
* Le traitement des 84916 images nécessite d'utilisation de générateurs. Ces derniers sont à customiser manuellement afin de permettre une gestion en batch des données textuelles et d'images pour le modèle de fusion.
* La création d'un modèle de fusion a été une tâche ardue, principalement pour la gestion des entrées sous forme de générateurs.

___
<!--
_header: 'Perspectives' 
-->
 
![bg right:60% width:70em](
https://global.fr.shopping.rakuten.com/wp-content/uploads/2020/05/rak-monde-bottom-img.png)

![width:100px](https://oxygentogo.com/wp-content/uploads/2017/05/blockquote-300x198.png)

Nous continuons de croire que le monde numérique a le potentiel d'améliorer la vie de chacun d'entre nous. Oubliez la peur. Adoptez l'optimisme.
 
 ***Hiroshi Mikitani** – Fondateur et CEO de Rakuten*



----
<!--
_header: 'Perspectives' 
-->


* Ajout d'autres modèles au modèle de fusion.
* Ajout d'autres modèles au modèle de fusion.
* Uniformisation des données dans le code. Actuellement, des dataframes Pandas, des tableaux Numpy, des générateurs d'images fonctionnent ensemble. Tout pourrait être géré autour d'un seul type de données, comme les tf.data.DataSet.
* Changement de la couche d'embedding ou création d'un modèle parallèle. Le modèle de texte par exemple pourrait être doté d'une couche d'embedding pré-entrainée, par exemple celle issue de CamemBERT. 
![bg right:45% ](
https://static9.depositphotos.com/1101919/1123/i/450/depositphotos_11238831-stock-photo-innovation-idea.jpg)


----

<!--
_header: '' 
-->
Le projet ![height:35px](https://upload.wikimedia.org/wikipedia/commons/thumb/0/0c/Logo_rakuten.jpg/1200px-Logo_rakuten.jpg) a été très intéressant, car complexe et faisant appel à des notions avancées mêlant le traitement de textes et le traitement d'images. 

L'exploration de données, le travail de groupe, les différentes implémentations et sprints ont fait de ce projet un projet répondant, nous l'espérons, aux besoins d'une entreprise.
![bg left:50% ](https://static.vecteezy.com/system/resources/previews/006/161/114/large_2x/conclusion-word-on-red-keyboard-button-free-photo.jpg)

---
<!--
_header: 'Annexe : Machine Learning  / Text' 
-->
* Catégorie `10` (Livre d'occasion) souvent confondue avec `2705` (Livre neuf) et `2403` (Revue) 
* Catégorie `40` (Jeu console) souvent confondue avec `10` (Livre occasion)  et `2462` (Jeu oldschool)
* Catégorie `1280` (Déguisement) souvent confondue avec `1281` (Boîte de jeu) et `1140` (Figurine)


![bg right:48% width:75%](images/models_ml_text.jpg)

___
<!--
_header: 'Annexe : Machine Learning / Image' 
-->
# Machine Learning / Image
| Classifier | Acc. | Precision weighted | Recall weighted | F1 weighted |
|------------|----------|--------------------|-----------------|-------------|
| LogReg     | 0.18     | 0.16               | 0.18            | 0.16        |
| RF         | 0.12     | 0.04               | 0.12            | 0.04        |
| KNN        | 0.18     | 0.16               | 0.18            | 0.16        |
| SVC        | 0.18     | 0.17               | 0.18            | 0.17        |
| GradBoost  | 0.09     | 0.08               | 0.09            | 0.06        |

![bg right:39% width:100%](../notebooks/images/SVCHeatmap.png)
___
<!--
_header: 'Annexe : Les modèles / Deep learning / Text ' 
-->
![bg width:60%](../notebooks/images/texts/epoch_accuracy.png)

___
<!--
_header: 'Annexe : Les modèles / Deep learning / Image ' 
-->
 |Model                        |Accuracy                 | Val accuracy    | 
|-----------------------------|-------------------------|-----------------|
| VGG16                       |0.50                     |0.49             | 
| ResNet                      |0.16                     |0.18             | 
| MobileNet                   |0.87                     |0.47             |

![bg right width:100%](../notebooks/images/images/epoch_accuracy_vgg16.png)