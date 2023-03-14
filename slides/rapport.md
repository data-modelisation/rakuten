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
___
<!--
_header: 'Description des données ' 
-->
<style scoped>
table {
  font-size: 14px;
}
table tr:nth-child(1) td:nth-child(1), 
table tr:nth-child(1) td:nth-child(2), 
table tr:nth-child(2) td:nth-child(5), 
table tr:nth-child(2) td:nth-child(6),
table tr:nth-child(4) td:nth-child(5), 
table tr:nth-child(4) td:nth-child(6),
table tr:nth-child(8) td:nth-child(5), 
table tr:nth-child(8) td:nth-child(6) 
{ 
  background: #B8CEC2;  
}

table tr:nth-child(2) td:nth-child(1), 
table tr:nth-child(2) td:nth-child(2),
table tr:nth-child(3) td:nth-child(1), 
table tr:nth-child(3) td:nth-child(2),
table tr:nth-child(4) td:nth-child(1), 
table tr:nth-child(4) td:nth-child(2), 
table tr:nth-child(1) td:nth-child(3), 
table tr:nth-child(1) td:nth-child(4),
table tr:nth-child(1) td:nth-child(5), 
table tr:nth-child(1) td:nth-child(6),
table tr:nth-child(3) td:nth-child(5), 
table tr:nth-child(3) td:nth-child(6),
table tr:nth-child(9) td:nth-child(5), 
table tr:nth-child(9) td:nth-child(6)
{ 
  background: #2C4F71;
  color: white; 
}

table tr:nth-child(5) td:nth-child(1),
table tr:nth-child(5) td:nth-child(2),
table tr:nth-child(6) td:nth-child(1),
table tr:nth-child(6) td:nth-child(2), 
table tr:nth-child(7) td:nth-child(1),
table tr:nth-child(7) td:nth-child(2), 
table tr:nth-child(8) td:nth-child(1),
table tr:nth-child(8) td:nth-child(2), 
table tr:nth-child(9) td:nth-child(1),
table tr:nth-child(9) td:nth-child(2),
table tr:nth-child(3) td:nth-child(3),
table tr:nth-child(3) td:nth-child(4)
{ 
  background: #F8E6CE; color: #2C4F71; 
}
table tr:nth-child(5)  td:nth-child(3),
table tr:nth-child(5)  td:nth-child(4),
table tr:nth-child(6)  td:nth-child(3),
table tr:nth-child(6)  td:nth-child(4),
table tr:nth-child(7)  td:nth-child(3),
table tr:nth-child(7)  td:nth-child(4),
table tr:nth-child(8)  td:nth-child(3),
table tr:nth-child(8)  td:nth-child(4),
table tr:nth-child(5)  td:nth-child(5),
table tr:nth-child(5)  td:nth-child(6)
{ 
  background: #EBC8B4;  color: green;
}

table tr:nth-child(6) td:nth-child(5),
table tr:nth-child(6) td:nth-child(6),
table tr:nth-child(7) td:nth-child(5),
table tr:nth-child(7) td:nth-child(6),
table tr:nth-child(9) td:nth-child(3),
table tr:nth-child(9) td:nth-child(4)
{ 
  background:  #D77a61; color: white; 
}

table tr:nth-child(2) td:nth-child(3),
table tr:nth-child(2) td:nth-child(4),
table tr:nth-child(4) td:nth-child(3),
table tr:nth-child(4) td:nth-child(4)
{ 
  background:  #DBD3D8;
}
</style>


![bg right:46% height:50%](../notebooks/images/images_category.png)

|Cat. | Code et libellé|Cat.| Code et libellé|Cat.| Code et libellé|
|---:|------------|----:|------------|-------------------:|------------|
| ![height:30px](https://www.icone-png.com/png/40/39859.png)   | 10 - Livre d'occasion   | ![height:30px](https://cdn-icons-png.flaticon.com/512/1223/1223280.png) | 1300 - Jouet Tech     |  ![height:30px](https://cdn-icons-png.flaticon.com/512/13/13282.png) | 2280  - Affiche       |
|  ![height:30px](https://cdn-icons-png.flaticon.com/512/25/25428.png)   | 40 - Jeu Console        | ![height:30px ](https://cdn-icons-png.flaticon.com/512/6824/6824500.png)  | 1301 - Chaussette     |  ![height:30px](https://www.icone-png.com/png/40/39859.png) | 2403 - Revue         |
|   ![height:30px](https://cdn-icons-png.flaticon.com/512/25/25428.png)|  50 -  Accessoire Console  |  ![height:30px](https://cdn-icons-png.flaticon.com/512/138/138409.png) | 1302 - Gadget         | ![height:30px](https://cdn-icons-png.flaticon.com/512/25/25428.png) | 2462 - Jeu oldschool |
| ![height:30px](https://cdn-icons-png.flaticon.com/512/25/25428.png)   | 60 - Tech               |  ![height:30px](https://cdn-icons-png.flaticon.com/512/6824/6824500.png)| 1320 - Bébé           |  ![height:30px](https://www.icone-png.com/png/40/39859.png)  | 2522 - Bureautique   |
|![height:30px](https://cdn-icons-png.flaticon.com/512/138/138409.png)    | 1140 -  Figurine          |   ![height:30px](https://cdn-icons-png.flaticon.com/512/165/165674.png)| 1560 - Salon          |  ![height:30px](https://cdn-icons-png.flaticon.com/512/165/165674.png) | 2582 - Décoration    |
| ![height:30px](https://cdn-icons-png.flaticon.com/512/138/138409.png) |  1160 - Carte colllect.   | ![height:30px](https://cdn-icons-png.flaticon.com/512/165/165674.png) | 1920 - Chambre        | ![height:30px](https://cdn.pixabay.com/photo/2017/10/24/11/53/tools-2884303_960_720.png) |  2583 - Aquatique     |
| ![height:30px](https://cdn-icons-png.flaticon.com/512/138/138409.png)  |  1180 - Jeu Plateau        | ![height:30px](https://cdn-icons-png.flaticon.com/512/165/165674.png)    | 1940 - Cuisine        |  ![height:30px](https://cdn.pixabay.com/photo/2017/10/24/11/53/tools-2884303_960_720.png) | 2585 - Soin et Bricolage |
| ![height:30px](https://cdn-icons-png.flaticon.com/512/138/138409.png)   |  1280 - Déguisement        |   ![height:30px](https://cdn-icons-png.flaticon.com/512/165/165674.png) | 2060 - Chambre enfant | ![height:30px](https://www.icone-png.com/png/40/39859.png)  | 2705 - Livre neuf    |     
|  ![height:30px](https://cdn-icons-png.flaticon.com/512/138/138409.png)  | 1281 - Boite de jeu       | ![height:30px](https://cdn.pixabay.com/photo/2017/10/24/11/53/tools-2884303_960_720.png)  | 2220 - Animaux        |   ![height:30px](https://cdn-icons-png.flaticon.com/512/25/25428.png)| 2905  - Jeu PC       |   

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
section p, li {
  font-size: 17px;
}

</style>
L'exemple de transformations appliquées : 
* `designation` : Une table très jolie! 
* `description` : <ul><li>\&#43;Dimensions : 60 x 33 cm</li><ul>

| Etape                                                 |     Résultat                                   | 
| :----- | :----------------------------------------------- | 
| Fusion de deux colonnes                               | Une table très jolie! <ul><li>\&#43;Dimensions : 60 x 33 cm</li></ul> | 
| Détection de la langue  et traduction en français        | Une table très jolie! <ul><li>\&#43;Dimensions : 60 x 33 cm</li></ul> | 
| Suppression les balises html                          | Une table très jolie! Dimensions : 60 x 33 cm  | 
| Suppression des caractères non alpha-numériques          | Une table très jolie Dimensions x cm           |
| Passage en minuscules                                  | une table très jolie dimensions x cm           |
| Supression des accènts                                              | une table tres jolie dimensions x cm           |
| Les mots d'un caractère                               | une table tres jolie dimensions cm             |
| Suppression des *stopwords*                           | table tres jolie dimensions cm                 | 
| Extraction de la racine des mots                      | tabl tres jol dimens cm                        | 
| Vectorisation du texte via un `Tokenizer`             | [6, 1, 2, 4, 5 ]                               | 
___

<!--
_header: 'Préparation des données / Images' 
-->
__Generateur d'image__:
* streaming per batch : les images sont transmises sous de batchs ce qui évite de traiter l'ensemble des données d'un coup
* rédimensionnement en taille `224x224 px`
* application de la fonctionne `preprocess_input` spécifique pour chaque modèle 

![bg right width:80%](../notebooks/images/rescale.png)
___
<!--
_header: 'Les modèles / Deep learning / Text ' 
-->
![bg height:80%](images/models_dl_text.jpg)

___
<!--
_header: 'Les modèles / Deep learning / Image ' 
-->
![bg width:90%](diagram/image_schema_MobileNetv2.drawio.png)
___
<!--
_header: 'Les modèles / Deep learning / Fusion ' 
-->

![bg right width:90%](../notebooks/images/fusion_methodology.png)

Un schéma simplifié du fonctionnement de concaténation.

* concaténation est faite sur les avant-dernières couches de deux modèles. 
* les autres couches sont *freezées*. 
* couches denses completent la fusion pour obtenir une classification sur 27 classes. 
___
<!--
_header: 'Les modèles / Deep learning / Fusion ' 
-->
![bg width:43%](images/models_dl_text.jpg)
![bg width:33%](images/models_fusion.jpg)

___
<!--
_header: 'Analyse du meilleur modèle' 
-->
<style scoped>
section p, li {
  font-size: 16px;  
}
header {
  padding-right:2px;
  margin: 0;
}
</style>
![bg right:75% height:76%](images/fusion_crosstab.jpg)
Pas d'impacte sur les performances réduites du modèle d'image.
  * Toutes les catégories dépassent le score de 54% et 
  * Une catégorie sur trois dépasse le score de 90%

Le modèle concaténé s'aide du modèle d'image pour catégoriser les produits où le modèle de texte sous-performait : 
  * La catégorie 1080 (Jeu Plateau) gagne 25 points
  * La catégorie 2705 (Livre neuf) gagne 23 points
___

<!--
_header: 'Limites' 
-->
* Le traitement des 84916 images nécessite d'utilisation de générateurs.
* Disponibilité limité de ressources de calcul de type GPU ou TPU via Google Colab. 
* Coupures de lien entre Google Drive et Google Colab ont entraîné une grande perte de temps 
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
<style scoped>
section p, li {
  font-size: 18px;
  
}

</style>
###### Les modifications globaux : 
* Uniformisation des données dans le code. Actuellement, des dataframes Pandas, des tableaux Numpy, des générateurs d'images fonctionnent ensemble. Tout pourrait être géré autour d'un seul type de données, comme les tf.data.DataSet.



###### Le modèle de texte: 
- une couche d'embedding pré-entrainée, par exemple celle issue de CamemBERT. 
 
 

![bg right:45% ](
https://static9.depositphotos.com/1101919/1123/i/450/depositphotos_11238831-stock-photo-innovation-idea.jpg)


----
<!--
_header: 'Perspectives' 
-->
<style scoped>
section p, li {
  font-size: 18px;
  
}
</style>
###### Le modèle d'image :
- évolution traitement et preprocessing des images  
  * croping d'image 
  * augmentation des données via transformation 
- évolution de modèles testés : 
    * implimenter _Batch Normalization_,
    * entraîner des couches de model issue de transfer learning  
    * configurer differement les hyperparamétres 
    * entraînement des couches de model issue de transfer learning 
- analyse de patterns generés par les couches 
- test autres modèles avec autre taille des images en entrés 

###### Fusion 
- ajout d'autres modèles au modèle de fusion
- test un autre approche de la fusion :  utiliser un modèle pour identifier un group global et ensuite sous-group precis. Par exemple premiere model prédit un group "Livre" et deuxieme model predit "Nouveau" ou "Ancien".
![bg right:45% ](
https://media.istockphoto.com/id/863607936/fr/photo/pour-faire-la-liste-sur-bloc-note-avec-stylo-sur-le-bureau-et-le-caf%C3%A9.jpg?s=612x612&w=0&k=20&c=tkrDkcqQTHXCihN7VZghK9baToxSGtV1rjSgeHxdbNg=)
___
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


