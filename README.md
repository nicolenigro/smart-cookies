# Smart Cookies

## Description

## Creativity Metrics
### 18 Criteria (Ritchie, 2007)
Inspired by Ritchie’s creativity metrics, we incorporated functions to evaluate the recipes produced by our system in terms of:
* **Typicality**: is the recipe a recognizable example of the target genre (cookie recipes)?
	* recipe_typicality()
		* This function compares each recipe generated by the system to each recipe in the inspiring set. A higher typicality score means that a recipe is more similar to the recipes in the inspiring set.
* **Novelty**: how similar/dissimilar is the recipe to other recipes generated by the system?
	* recipe_novelty()
		* This function compares the recipes generated by the system to each other by determining which recipes have the most unique ingredients. This higher the novelty score, the more unique/novel ingredients a recipe contains. 
* **Quality**: is the recipe of value? will the ingredients in the recipe taste well together?
	* similarity_fitness()
		* This function uses the information in the npy files to evaluate the ‘similarity’ of all ingredients in the recipe to determine if the ingredients in the recipe ‘taste good’ together. A higher similarity score means that the recipe ingredients should taste better together. 

Our system uses similarity_fitness() and recipe_novelty() to rank recipes, giving each an overall novelty and quality score. This scoring methodology ranks recipes with ingredients that pair well together and are rarer higher. We also give each recipe a typicality score, but did not use this for calculating ranks because our crossover function guarantees that the recipes are typical enough to be considered a cookie. In our crossover, we take the essential ingredients from one of the parent cookies, so that all the recipes that we generate have dough that makes sense for a cookie. The typicality function helps us to see how close our generated recipes are to the inspiring set, but doesn’t reflect the quality/rank of a cookie recipe as well as our other two ranking functions.

Also, Ritchie emphasized the importance of an inspiring set because any creative system is based on some existing samples in one way or another. We intentionally created an inspiring set of 50 cookie recipes featuring all sorts of ingredients, with recipes featuring typical cookie ingredients such as chocolate and oats but also recipes including more unique ingredients such as cumin and zucchini. With a wide variety of ingredients, our system is able to create unique and novel cookie recipes while maintaining the key ingredients of cookies.

### Four PPPPerspectives (Rhodes, 1961; Jordanous, 2016)  
Jordanous suggested that there are four perspectives that a successful system should incorporate--product, producer, process, and press. Above, we focused on the ‘product’--the quality of the cookie recipes generated by the system. Our system can also be evaluated in terms of process--how does our system connect to the way that humans go through the creative process. 

Our system’s process can be evaluated through the idea of Ventura’s odyssey. Different parts of the system can be categorized as being in different stages of the odyssey. The dough of each generated cookie recipe comes directly from the inspiring set to ensure that every cookie recipe has good dough--this aspect of our system is still in the memorization stage. Inspired by the generalization phase of Ventura’s odyssey, we categorize all ingredients as either essential or nonessential, which impacts how the ingredient is added or removed from a recipe. We also use our evaluation functions to create a bias for recipes with ingredients that ‘go well’ together. Our fitness functions and rankings bring our system into the filtration phase--with these functions we are able to evaluate our cookie recipes and determine which recipes will go on to the next generation. This filtration of cookie recipes based on similarity fitness and uniqueness are a major part of our system’s creative process. Although we have a knowledge base with information about ingredient types and ingredient similarity, we don’t consider our system to be entirely at the inception stage of Ventura’s odyssey. Part of the inception phase is  “inject[ing] knowledge into a computationally creative system without leaving the injector’s fingerprints all over the resulting artifact.” Most of our domain knowledge is hard coded in--to improve on our system in the future, we could increase the size of our inspiring set more, and try to create a wider knowledge base. 

## How to Set Up and Run the Code
1. Open the terminal.
2. Change your directory to the folder you want to store this code in.  
```
$ cd Documents/GitHub
```
3. Clone this repository onto your computer with the following line:  
```
$ git clone https://github.com/nicolenigro/lets-get-cooking.git
```
4. Change your directory to the folder for this project.  
```
$ cd smart-cookies
```
5. Install all the necessary packages with the following commands:
```
$ pip3 install numpy
$ pip3 install beautifulsoup4
$ pip3 install requests
```
6. Type and enter the following line into the terminal to run the program.  
```
$ python3 cookie_generator.py
```

## Authors
* Ahmed Hameed
* Adrienne Miller
* Nicole Nigro

## Inspiring Set
We created an inspiring set of cookie recipes using recipes from these sites:
* https://cooking.nytimes.com
* https://sallysbakingaddiction.com
* https://www.bonappetit.com
* https://www.biggerbolderbaking.com
* https://www.epicurious.com
* http://thesweetandsimplekitchen.com
* https://kirbiecravings.com
* https://www.kingarthurbaking.com
* https://www.allrecipes.com
* https://www.tasteofhome.com
* https://lilluna.com
* https://www.aspicyperspective.com
* https://www.cooking-therapy.com