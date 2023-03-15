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
@import url('https://fonts.googleapis.com/css2?family=Roboto&display=swap');
section {
  background-image: none;
  font-family:  'Roboto'; 
  font-weight: normal; 
  font-size: 1.4em;
  padding-top: 90px;
  padding-left: 40px;
  padding-right: 40px;
   
}
/* https://github.com/marp-team/marpit/issues/271 */
section::after {
  #font-weight: 700;
  font-size: 0.6em;
  content: attr(data-marpit-pagination) '/' attr(data-marpit-pagination-total);
}
section.lead h1, section.lead h2, section.lead h3{
  color: #bf0000;
  text-align: left;
}
header {
  background-color: #bf0000;
  color: #fff;
  font-size: 1em;
  font-weight: 700;
  padding: 0.2em 2em 0.2em 2em;
  opacity:.5;
  border-radius: 15px;
}
footer {
  font-size: 0.6em;
}
container{
  display:flex;
}
col{
  flex:1
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

![bg left height: 100px](https://img.freepik.com/free-vector/ai-technology-brain-background-vector-digital-transformation-concept_53876-112224.jpg?w=826&t=st=1678478673~exp=1678479273~hmac=30056e96f26cfed14acb6d22fe55d7329c23fe2998a8ee425cc206b63a812474)

![top:0px width:600px](https://challengedata.ens.fr/logo/public/RIT_logo_big_YnFAcFo.jpg) 

![height:60px right](https://i0.wp.com/datascientest.com/wp-content/uploads/2022/03/logo-2021.png?w=429&ssl=1)


__Participants :__
Olga TOLSTOLUTSKA
Mohamed BACHKAT
Charly LAGRESLE


__Mentor :__ Manu POTREL
__Promotion:__ DST Bootcamp DEC22

___


<!--
_header: 'Contexte' 
-->

## Rakuten

* Site de e-commerce avec 1.3 milliards d'utilisateurs
* Suggestions de recherche et recommandations pour l'utilisateur
* Classification des produits nécessaire
* Manuellement impossible

## Objectifs

Prédire la catégorie d'un produit sur la base de son **titre**, sa **description** et de son **image**

1 + 1 = 3 ... Un **modèle de texte**, un **modèle d'image** et un **modèle de fusion** 

___

<!--
_header: 'Présentation des données' 
-->


* **84 916** observations
* **27** catégories à déterminer 
* **0** données dupliquées 
* **Textes** 
  * Un produit est désigné par : `designation`   et `description` soit un titre et sa description
  * 35% de NaNs pour `description`
* **Images**
  * Une image couleur par produit
  * Peut comporter un support ou une mise en scène
  * Taille `500x500px` en JPG

![bg right:46% height:50% opacity:.7](../notebooks/images/images_category.png)
___
<!--
_header: 'Exploration des données | Cibles' 
-->

## Données déséquilibrées
* 27 catégories labelisées
* 7 domaines différents trouvés (non labelisés)
* Sur-représentation de la classe `2583`
* Sous-représentation des classe `60`, `1320` et `2220`

## Challenge probable
&rarr; Les modèles auront probablement (comme nous) du mal à distinguer les catégories de produits du même domaine

![bg right height:90%](./images/categories.svg)

___
<!--
_header: 'Exploration des données / Textes' 
-->

## Extraction de données quantitives
* Nombre de mots très variable

## Langues
* Détection de la langue
  * 81% français
  * 14% anglais
  * Traduction


![bg width:50%](../notebooks/images/words.png)

![bg left:59%](../notebooks/images/common_words.png)

___

<!--
_header: 'Exploration des données / Images' 
-->

## Analyse de canaux
* TODO

## Analayse de la variance
* Masque de variance très net sur les bordures des images
* Possibilité de rogner les images de 20% sans perdre trop d'information

![bg right:40% height:90%](./images/channels.svg)
___
<!--
_header: 'Préparation des données / Textes' 
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

| Étape                                                 |     Résultat                                   | 
| :----- | :----------------------------------------------- | 
| Fusion de `description` et `designation` | Une table très jolie! <ul><li>\&#43;Dimensions : 60 x 33 cm</li></ul> | 
| Détection de la langue  et traduction en français        | Une table très jolie! <ul><li>\&#43;Dimensions : 60 x 33 cm</li></ul> | 
| Suppression les balises html                          | Une table très jolie! Dimensions : 60 x 33 cm  | 
| Suppression des caractères non alpha-numériques          | Une table très jolie Dimensions x cm           |
| Passage en minuscules                                  | une table très jolie dimensions x cm           |
| Suppression des accents                                              | une table tres jolie dimensions x cm           |
| Suppression des mots d'un caractère                               | une table tres jolie dimensions cm             |
| Suppression des *stopwords*                           | table tres jolie dimensions cm                 | 
| Extraction de la racine des mots                      | tabl tres jol dimens cm                        | 
| Vectorisation TF-IDF du texte via un `Tokenizer`             | [6, 1, 2, 4, 5 ]                               | 
___

<!--
_header: 'Préparation des données / Images' 
-->
__Générateur d'images__:
* Streaming per batch : les images sont transmises sous de batchs ce qui évite de traiter l'ensemble des données d'un coup
* Redimensionnement en taille `224x224 px`
* Application de la fonction `preprocess_input` spécifique à chaque modèle 

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
![bg right width:90%](../src/models/fusion/fusion_mobilenet_simple_224/model.png)


Un schéma simplifié du fonctionnement de concaténation.

* concaténation est faite sur les avant-dernières couches de deux modèles. 
* les autres couches sont *freezées*. 
* couches denses completent la fusion pour obtenir une classification sur 27 classes. 

___
<!--
_header: 'Analyse du meilleur modèle 1/2' 
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
![bg right width:60%](./images/f1score_all.png)

## Analyse des *weigthed f1-scores*
 
  * Toutes les catégories dépassent le score de 55%
  * Une catégorie sur trois dépasse le score de 90%
  * Les catégories `10`,  `40` et `2705`  très impactées par la fusion
  * Au final : *weighted f1-score* 82.2 %

  Le modèle concaténé s'aide du modèle d'image pour catégoriser les produits où le modèle de texte sous-performait : 
  * La catégorie `10` <span style="font-weight:bold;color:#c7e5d7ff">Livre neuf</span> gagne 15 points
  * La catégorie `2705` <span style="font-weight:bold;color:#c7e5d7ff">Livre occasion</span> gagne 18 points
___
<!--
_header: 'Analyse du meilleur modèle 2/2' 
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
![bg right:60% width:100%](./images/fusion_crosstab.svg)

## Analyse des erreurs > 10%
* <span style="font-weight:bold;color:#c7e5d7ff">Livres</span> : `10`, `2080`, `2403` et `2280`
* <span style="font-weight:bold;color:#ffcc5cff">Jouets</span> : `1080`, `1280` et `1281`
* <span style="font-weight:bold;color:#ff6f69ff">Mobilier</span>: `2582` et `1560`

Nous nous attendions à avoir des erreurs au sein de produits du même domaine
___

<!--
_header: 'Challenges' 
-->
* Le traitement des 84916 images nécessite d'utilisation de générateurs.
* Disponibilité limité de ressources de calcul de type GPU ou TPU via Google Colab. 
* Coupures de lien entre Google Drive et Google Colab ont entraîné une grande perte de temps 
* La création d'un modèle de fusion a été une tâche ardue, principalement pour la gestion des entrées sous forme de générateurs.


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
_header: 'Conclusion' 
-->
 
 Merci à tous pour ces 3 mois très intenses et riches ! 

[Lien vers le Streamlit](localhost:8080)


![bg right:60% width:70em](
https://global.fr.shopping.rakuten.com/wp-content/uploads/2020/05/rak-monde-bottom-img.png)

![width:80px](https://oxygentogo.com/wp-content/uploads/2017/05/blockquote-300x198.png)

Nous continuons de croire que le monde numérique a le potentiel d'améliorer la vie de chacun d'entre nous. Oubliez la peur. Adoptez l'optimisme.
 
 ***Hiroshi Mikitani** – Fondateur et CEO de Rakuten*



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

___
<!--
_header: 'Annexe Exploration des données / Target' 
-->

## Notions
* cec
* cze

![bg height:50%](../notebooks/images/imbalanced.png)

___


___
<!--
_header: 'Annexe | Choix de la métrique' 
-->

## Notions
* Connaissance du métier : une erreur de classification n'est pas fatale
* Labelisation : comment a-t-elle été effectuée
* Jeu de données déséquilibré : dû à une survente ou à des difficultés à classer ces produits
* Forte tendance à l'*overfitting*


**&rarr;** Choix de la métrique : *f1 weigthed score* pour un bon équilibre entre *accuracy* et *recall*

## Remarques

* Modèle aléatoire : score de 3.7% en moyenne
* Une métrique personnalisée aurait pu être créée

___


