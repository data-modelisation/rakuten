
# Markdown présentation   
Marp est un outil qui aide à créer une présentation avec Markdown et ensuite générer un Powerpoint ou PDF.

## Aperçu
Le plugin "Marp for VS Code" aide à visualiser une présentation créée.  
Pour l'installer il faudrait d'aller dans le menu de VS code 

File -> Prefrences -> Extensions  

rechercher "Marp for VS Code" er appuyer sur le bouton install. 


Le visuel s'affiche en temps réel de future présentation à partir de code ajouté.

## Syntax 
Le fichier de la presentation a une extension `.md` et commence par une directive `marp: true` pour indiquer qu’il s’agit d’un document Marp et pas Markdown.
La balise « — » est utilisée  pour indiquer les nouvelles slides. 

Par exemple le code suivant génère deux slides :
```sh
---
marp: true
---
Slide 1 
---
Slide 2 
```

## Les directives globales

Les directives globales sont appliquées pour tous les slides.
Par exemple :

```
marp: true
theme: gaia
footer: projet Rakuten    
```
## Les directives locales 
Les directives locales  sont appliquées pour le slide et peuvent redéfinir les directive globales pour le slide spécifique. Par exemple:
```html
<!--
_header: 'Context' 
-->
```

## Image syntax

Pour ajouter une image avec une taille spécifique : 
```
![width:100px height:100px](image.png)
```

Pour afficher une image en arrière plan: 
```
![bg right:30%] (https://… /image.jpg)
```
On peut afficher plusieurs images en arrière plan :
```
![bg] (https://… /image1.jpg)
![bg] (https://… /image2.jpg)
```

## Style 
Le style de la présentation peut être enrichi grâce aux règles css. Par exemple : 
```css 
/* @theme yout-theme */
@import 'default';

section {
  width: 960px;
  height : 720 px
}

h1 {
	font-size : 30px;
  color: #c33;
}
```


# Export to Powerpoint et PDF 
Pour pouvoir effectuer l'export il est nécessaire d'installer la commande  marp.  


## Install Windows 

```sh
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
iwr -useb get.scoop.sh | iex
scoop install marp
```

## Les commandes  
Pour génèrer un powerpoint exécutez le code suivant depuis la racine du projet : 

```
marp --pptx slides/rapport.md  --allow-local-files
```

Le parametre  `--allow-local-files` doit être specifié pour inclure les fichier locaux dans la presentation. 

Pour génèrer un pdf exécutez le code suivant depuis la racine du projet :

```
marp --pdf slides/rapport.md  --allow-local-files
```

