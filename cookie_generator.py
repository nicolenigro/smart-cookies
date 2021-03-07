"""
PQ2: Smart Cookies
CSCI 3725
Team OOG: Ahmed Hameed, Adrienne Miller, Nicole Nigro
Version Date: 3/7/21

Description: This system generates cookie recipes.
"""

"""
Outline
-Take Ingredient and Recipe classes from PQ1
    *Ingredient
        -Unit (i.e. cups, teaspoons)
        -Flavor similarity (Ingredient.similarity to compare to other Ingredients)
        -Type of ingredient: flour, sugar, shortening
        -Essential ingredients vs mix-in (T/F)
    *Recipe
        -Special naming function
            *pick from list of adjectives, pick one of the mix-ins, "cookies"
                *alliteration name
        -Scores/weights
        -Stick to basic recipe's cooking times/steps
        -Keep instructions relatively constant
        -Origin attribute: if Recipe has x Ingredients, then it's ____ origin
            *incorporate to recipe title
-Generation
    -Dictionary? of ingredients and average amount <- just for essential ingredients
        *check how far the amount is from the average

Inspiring Set
- look through recipes Harmon listed, create .txt files for them, add to input (start w 5)
    -> web scrape eventually?
    *use to find out basic ingredients of a cookie
    *look for constants like time and temp for this week
    *use grams

Knowledge Base
-Ingredient pairing? (might be for next week)
-Base ingredients for every cookie, focus on mix-ins
    *what mix-ins pair well with interesting essential ingredients (spices)
    *numpy weights for delete/add mutations: get rid of worst pairing, add best pairing
    *change amount just for mix-ins, having smaller amt of values an essential ingredient can change to
-Update mutations: look at input file to determine an appropriate range of flour, sugar
-List of adjectives for naming function
""