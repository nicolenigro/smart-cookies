"""
PQ2: Smart Cookies
CSCI 3725
Team OOG: Ahmed Hameed, Adrienne Miller, Nicole Nigro
Version Date: 3/11/21

Description: This system generates cookie recipes.

Dependencies: glob, numpy, os, random
"""

"""
Week of March 8th --> 
- naming function in Recipe class [X]
- updating readings in files [X]
- mutations: incorporate flavor pairings[X]
    - need to fine tune this --> we are getting super weird stuff 
- crossover --> only do on non essentials[X]
- fitness function is average compatability of nonessential ingredients 

Meetings:
- Nicole and Ahmed on March 8th, 1:30-2:30 PM, to work on 
    - sorting of Ingredients in Recipe based on essential and non-essential ingredients.
        - Want compound ingredients like "brown sugar" to be read in as an essential ingredient (use .contains())
    If there is still time left over then:
    - 

Outline
-Take Ingredient and Recipe classes from PQ1
    *Ingredient
        -Unit (i.e. cups, teaspoons) [X]
        -Flavor similarity (Ingredient.similarity to compare to other Ingredients) [X]
        -Type of ingredient: flour, sugar, shortening
        -Essential ingredients vs mix-in (T/F) [X]
    *Recipe
        -Special naming function [X]
            *pick from list of adjectives, pick one of the mix-ins, "cookies" [X]
        
        -Scores/weights
        -Stick to basic recipe's cooking times/steps
        -Keep instructions relatively constant
        -Origin attribute: if Recipe has x Ingredients, then it's ____ origin
            *incorporate to recipe title
        -Normalize (take into account yield)
        --> We could have a mutation function for each TYPE of base ingredient, keeping amounts constant
                For example:
                        * mutate_flour(flour_name) - swaps the given flour ingredient for another type of flour
                                                    all purpose flour --> whole wheat flour
                        * mutate_sugar(sugar) - swaps the sugar
                        * mutate_shortening(short) - swaps the shortening
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
    *consider all butters as a core ingredient
    *numpy weights for delete/add mutations: get rid of worst pairing, add best pairing
    *change amount just for mix-ins, having smaller amt of values an essential ingredient can change to
-Update mutations: look at input file to determine an appropriate range of flour, sugar
-List of adjectives for naming function

Plan:
Inspring Set of plain dough recipes
randomly select a dough recipe and use its measurements
randomly select the first mix in
decide the future mix ins depending on flavor pairing

TO DO LIST AS OF MARCH 10 --
    - figure out categories to get rid of weird ingredients ( in flavor_pairing.py ) **
        - can we add fun mixins to ingredient file? 
    - edit input recipes so that ingredient names match flavor data set **
    - (Not urgent, but should be done before submit code) clean up bugs and exception stuff  * 
    (when are ingredients aren't in database --> how do we deal)
    - (Not urgent): Generate some random recipes based using ingredient names from flavor data set so that
        there are no conflicts in naming. Random recipes will have almost identical base ingredients but mix-ins
        will vary

EVALUATION
-similarity_fitness() is 1 form of evaluation
-recipe_uniqueness() is 2nd form of evaluation

Week of 3/15 to do list
-recipe_uniqueness() [X]
    *get more points based on uniqueness of an ingredient in the final recipe set
    *1/occurnece of an ingredient
        *cinnamon will have a lower score than matcha
-calculate_rank() function <- do this Generator [X]
    *sum of uniqueness and similarity
    *higher the value, higher the rank
    *order based on value
    *for an output of 10 recipes, ranks them #1, #2, #3 ... best recipe
    *write as final line of recipe (after instructions) ---> ***** Do we need this **(* )
-fix the instructions to be more inclusive of mix-ins
    *preheat
    *dry ingredients
    *wet ingredients
    *bake
-going through inspring set to paraphrase ingredients for numpy 
-keep instructions from recipe that is being used as base in crossover
"""

import string
import time
import glob
import numpy as np
import os
import random
from flavor_pairing import similarity, pairing
from bs4 import BeautifulSoup
import requests
import json

ESSENTIAL_INGREDIENTS = ["sugar", "butter", "flour", "egg", "eggs", "yolk", "yolks", "baking soda", "baking powder", "salt"]

class Ingredient:
    def __init__(self, name, quantity, unit="grams", essential=False):
        """
        Initializes an ingredient with a name and a quantity.
        Args:
            name (str): name of the ingredient
            quantity (float): amount of the ingredient, measured in oz
            unit (str): unit of measurement
            essential (bool): if it is an essential ingredient or not
        """
        self.name = name
        self.quantity = abs(quantity)
        self.unit = unit
        self.essential = essential
        self.mark_essential_ingredient()

    def mark_essential_ingredient(self):
        """
        Uses the essential ingredients list to marks which Ingredients are considered "essential" 
        for a cookie recipe
        Args:
            None
        Return:
            None
        """
        for item in ESSENTIAL_INGREDIENTS:
            if (item in self.name):
                self.essential = True

    def __str__(self):
        """
        Method for printing out an Ingredient object 
        Args:
            None
        Return:
            output (str): The string output representing a Ingredient object
        """
        return f"{self.name} {self.quantity} {self.unit}"


class Recipe:
    def __init__(self, name, ingredients, instructions):
        """
        Initializes a recipe with a name, a list of Ingredients that make up the recipe, and the
        corresponding instructions.
        Args:
            name (str): The name of the recipe (make creative later on) 
            ingredients (list): List of Ingredient objects
            instructions (str): A string with all the baking instructions
        """
        self.name = name
        self.ingredients = ingredients
        self.instructions = instructions

    def similarity_fitness(self):
        """
        
        Args:
            None
        Return:
            fitness_score (int): the fitness score of a recipe
        """
        fitness_score = 0
        non_essentials = self.get_non_essentials()

        for i in range(len(non_essentials)):
            ingred1 = non_essentials[i].name
            for j in range(i, len(non_essentials)):
                ingred2 = non_essentials[j].name
                # THIS TRY STATMENT IS BAD
                try:
                    sim = similarity(ingred1, ingred2)
                except:
                    sim = 0.5
                fitness_score += sim

        return fitness_score 

    def get_non_essentials(self):
        """ 
        Returns a list of all the non-essential ingredients in ingredient_list
        Args:
            None
        Return:
            non_essentials (list) - The list of non-essential ingredients
        """
        non_essentials = []
        # extract list of nonessential ingredients
        for ingredient in self.ingredients:
            if not ingredient.essential:
                non_essentials.append(ingredient)
        return non_essentials

    def add_ingredient(self, new_ingredient):
        """
        Adds a new Ingredient object to the list of Ingredients.
        Args:
            new_ingredient (Obj): Ingredient to add
        Return:
            None
        """
        self.ingredients.append(new_ingredient)

    def delete_ingredient(self, ingredient_getting_deleted):
        """
        Deletes an ingredient from the recipe
        Args:
            ingredient_getting_deleted (Obj): Ingredient to delete
        Return:
            None
        """
        self.ingredients.remove(ingredient_getting_deleted)

    def combine_duplicates(self):
        """
        Iterates through ingredients list and creates dictionary mapping ingredient name to ingredient 
        quantity, accounting for duplicates. Uses that dictionary to create a duplicate free ingredient list
        and update self.ingredients.
        Args:
            None
        Return:
            None
        """
        new_ingredients_dict = dict()

        # iterate through self.ingredients,
        for ingredient in self.ingredients:
            if ingredient.name in new_ingredients_dict:
                # this ingredient already found, increase quantity
                new_ingredients_dict[ingredient.name] += ingredient.quantity
            else:
                # new ingredient found
                new_ingredients_dict[ingredient.name] = ingredient.quantity

        new_ingredients = []
        # create list of ingredients with no duplicates
        for name, quantity in new_ingredients_dict.items():
            new_ingredient = Ingredient(name=name, quantity=quantity)
            new_ingredients.append(new_ingredient)

        self.ingredients = new_ingredients

    def name_recipe_helper(self, name_ingredient):
        """
        Given the name of an ingredient, uses describingwords.io to generate a list of common adjectives 
        associated with that word and a list of their occurrence score. 

        WARNING: Calling this function too many times will cause the server to kick you out resulting in an
                error. Please wait at least 5 seconds between runs. 

        Args:
            name_ingredient (str) - The name of the ingredient which will be looked up on describingwords.io
                                    and whose adjectives will be generated for
        Return:
            words (list) - A list of adjectives that are commonly associated with name_ingredient
            scores (list) - A score indicating a how often an adjective is used with that word                     
        """
        # format url and get html_doc object
        url = f"https://describingwords.io/for/{name_ingredient}"
        html_doc = requests.get(url=url).content
        soup = BeautifulSoup(html_doc, 'html.parser')

        # extract correctly tagged object and load it as list of dictionaries of words and their scores
        script_tag = soup.find(
            "script", id="preloadedDataEl", type="text/json")
        script_content = script_tag.contents
        related_words_str = (script_content[0])
        related_words_list = json.loads(related_words_str)["terms"]

        # loop through list, adding a related word and its score to lists
        words, scores = [], []
        for word_dict in related_words_list:
            words.append(word_dict["word"])
            scores.append(word_dict["score"])

        return words, scores

    def name_recipe(self):
        """
        Name a recipe with an adjective, ingredient name, cookies
        """
        non_essentials = self.get_non_essentials()
        # if no non essential ingredients --> recipe is basic
        if len(non_essentials) == 0:
            name_ingredient = "basic"
        else:
            # pick a random non essential ingredient
            name_ingredient = np.random.choice(non_essentials).name

        # get list of words and associated weights based on occurrenc
        related_words, weights = self.name_recipe_helper(name_ingredient)
        if len(related_words) == 0:
            adjective = "Uncool"
        else:
            # select adjective based on weighted scores from website
            adjective = ''.join(random.choices(related_words, weights, k=1))
        name = f"{string.capwords(adjective)} {string.capwords(name_ingredient)} Cookies"

        self.name = name
        return name

    def recipe_export(self, output_dir):
        """ 
        Given an output_dir, writes out contents of recipe to that file.
        Args: 
            output_dir (str): Name of file that will be written to
        Return:
            None
        """
        output_path = os.path.join(output_dir, self.name)
        f = open(output_path, "w")
        for ingredient in self.ingredients:
            if (round(ingredient.quantity, 2) < 0.01):
                line = f"{ingredient.quantity} grams {ingredient.name} \n"
                f.write(line)
            else:
                line = f"{round(ingredient.quantity,2)} grams {ingredient.name} \n"
                f.write(line)
        f.write("\n")
        f.write("Instructions")
        f.write(self.instructions)
        f.close()

    def __str__(self):
        """
        Method for printing out a Recipe object 
        Args:
            None
        Return:
            output (str): The string output representing a Recipe object
        """
        output = ""
        for item in self.ingredients:
            output += item.name + " " + (str)(item.quantity) + "\n"

        output += "\nInstructions"
        #output += self.instructions
        return output


class Generator:
    def __init__(self):
        """
        Initializes an Generator Object with an empty ingredient_names list (all ingredients in all recipes), 
        an empty recipe list (all recipes in current population) and a counter for new recipes (initalized at 1)
        Args: 
            None
        """
        self.ingredient_names = []  # all ingredients in population
        self.recipes = []
        self.new_recipe_count = 1
        self.default_mix_ins = []
        self.get_default_list()

    def read_files(self, input_directory):
        """
        Opens the files in the input folder, reading each file line by line to create Ingredient
        objects that make up Recipe objects. Adds each new ingredient encountered to list
        of all ingredients and adds each example Recipe to list of all recipes. 
        Args: 
            input_directory (st): name of input directory that recipe files are in
        Return: 
            None
        """
        for filename in glob.glob(input_directory):  # open each example recipe file
            ingredients = []  # intialize list of all ingredients in current recipe
            f = open(os.path.join(filename))

            input_string = ""
            for line in f.readlines():  # add each ingredient line in file to recipe
                if ("grams" in line):
                    # split line into list of form [quantity, ingredient]
                    split_line = line.rstrip().split(" grams ")
                    ingredient_name = split_line[1]
                    print(ingredient_name)
                    ingredient_quantity = round(float(split_line[0]), 2)
                    ingredient = Ingredient(
                        name=ingredient_name, quantity=ingredient_quantity)
                    ingredients.append(ingredient)
                input_string += line

                # add to list of all ingredients in example file set
                if ingredient_name not in self.ingredient_names:
                    self.ingredient_names.append(ingredient_name)

            split_input = input_string.split("Instructions")
            instructions = split_input[1]

            # create recipe with all ingredients in a file
            recipe = Recipe(name=filename, ingredients=ingredients, instructions=instructions)

            # save recipe to the list of recipes
            self.recipes.append(recipe)

    def crossover(self):
        """
        Weighted by recipe fitness, selects two 'parent' recipes to be crossed over. Then, 
        selects a random pivot index in each parent and generates a new child recipe. 

        Args: 
            None 
        Return: 
            child (Recipe): new Recipe object generated by crossing over two parents in the current population
        """
        # calculate fitness of each recipe
        recipes_fitness = [recipe.similarity_fitness() for recipe in self.recipes]
        parent1, parent2 = random.choices(population=self.recipes, weights=recipes_fitness, k=2)

        # crossover the non-essentials/mix ins of both parent recipes and combine w/ parent1 essentials
        non_essentials_parent1 = []
        essentials_parent1 = []
        for ingredient in parent1.ingredients:
            if ingredient.essential:
                essentials_parent1.append(ingredient)
            else:
                non_essentials_parent1.append(ingredient)

        non_essentials_parent2 = []
        for ingredient in parent2.ingredients:
            if not ingredient.essential:
                non_essentials_parent2.append(ingredient)

        pivot_1 = random.randint(0, len(non_essentials_parent1))
        pivot_2 = random.randint(0, len(non_essentials_parent2))

        # create subset of parent recipes that will be used in child
        subgroup_recipe1 = non_essentials_parent1[1:pivot_1]
        
        # create child recipe object
        subgroup_recipe2 = non_essentials_parent2[pivot_2:]

        child_name = "recipe" + str(self.new_recipe_count) + ".txt"
        child_ingredients = essentials_parent1 + subgroup_recipe1 + subgroup_recipe2
        child = Recipe(name=child_name, ingredients=child_ingredients, instructions=parent1.instructions)
        child.combine_duplicates()

        self.new_recipe_count += 1

        return child

    def mutation(self, recipe, prob_of_mutation):
        """
        Mutates a recipe by randomly selecting a mutation type(changing the ammount of an ingredient, switching an ingredient, 
        adding a new ingredient, or deleting an ingredient) and then executing that mutation.

        Args: 
            recipe (Recipe): recipe to be mutated 
            prob_of_mutation (float): probablity that a mutation will occur
        Return: 
            recipe (Recipe): either original recipe or recipe that has been mutated in one way. 
        """
        # probability that mutation will occur, if there isn't a mutation, just return recipe as is
        mutation = random.random() < prob_of_mutation
        if not mutation:
            return recipe

        # select random mutation type
        mutation_types = ["add", "delete"]
        current_mutation = random.choice(mutation_types)
        non_essentials = []

        for ingredient in recipe.ingredients:
            if not ingredient.essential:
                non_essentials.append(ingredient)

        if (current_mutation == "add"):
            # randomly ordered list of nonessentials
            random_order = random.choices(non_essentials, k=len(non_essentials))
            found_ingredient = False

            # loop through nonessential ingredients until you find one that works
            for i in range(len(random_order)):
                reference_ingredient = random_order[i]
                try:
                    good_pairings = pairing(reference_ingredient.name, 0.5)
                    good_ingredients, weights = good_pairings.items()
                    new_ingredient = random.choices(good_ingredients, weights=weights)
                except:
                    continue
                found_ingredient = True
                break

            if not found_ingredient:
                new_ingredient = random.choice(self.default_mix_ins)

            new_ingredient_amount = random.uniform(1, 50)  # FIX THIS
            recipe.add_ingredient(Ingredient(new_ingredient, new_ingredient_amount))

        elif (current_mutation == "delete"):
            # select random ingredient to remove from recipe
            ingredient_to_delete = random.choice(recipe.ingredients)
            recipe.delete_ingredient(ingredient_to_delete)

        recipe.combine_duplicates()
        return recipe

    def generate_population(self, mutation_prob):
        """
        Generates the new generation a new generation of recipes by calling the crossover function and 
        then mutating those new recipes. Selects next generation of recipes by sorting the previous and current
        generation by fitness, and taking the top 50% of each generation to create the new generation. 

        Args: 
            mutation_prob (float): the probability of mutation
        Return: 
            None
        """
        prev_generation = self.recipes
        next_generation = []
        num_children = len(self.recipes)

        # generate num_children number of children and mutate them
        for i in range(num_children):
            new_recipe = self.crossover()
            mutated_recipe = self.mutation(
                recipe=new_recipe, prob_of_mutation=mutation_prob)
            next_generation.append(mutated_recipe)

        # sort previous and next generation by their fitness
        sorted_previous = sorted(
            prev_generation, key=lambda recipe: recipe.similarity_fitness(), reverse=True)
        sorted_next = sorted(
            next_generation, key=lambda recipe: recipe.similarity_fitness(), reverse=True)

        # take top 50% of each generation to create the new recipe 'population'
        midpoint_index = len(sorted_previous) // 2
        new_population = sorted_previous[0:midpoint_index] + \
            sorted_next[0:midpoint_index]

        self.recipes = new_population

    def generate(self, num_generations, mutation_prob):
        """ 
        Performs the genetic algorithm for num_generations and returns the final recipes generated. 

        Args: 
            num_generations (int): number of new generations to be created before returning final recipe generation
            mutation_prob (float): the probability of mutation
        Return: 
            self.recipes (Recipe[]): final list of recipes resulting from several generations of crossover 
        """
        for i in range(num_generations):
            self.generate_population(mutation_prob)
        return self.recipes
        
    def get_default_list(self): 
        """
        """ 
        all_spices = pairing("vanilla", 0.00, cat="spice")
        all_herbs = pairing("vanilla", 0.00, cat="herb")
        all_nuts = pairing("vanilla", 0.00, cat="nut/seed/pulse")


        self.default_mix_ins += list(all_spices.keys())
        self.default_mix_ins += list(all_herbs.keys())
        self.default_mix_ins += list(all_nuts.keys())

        print(self.default_mix_ins)
    
    
    def recipe_uniqueness(self, recipes): 
        """ 
        Iterate over the recipes our system generated, keeping track of how often ingredients 
        appear. A higher uniqueness scores means more unique ingredients; a lower uniqueness
        score means mostly generic ingredients.
        Args:
            recipes (list): recipes generated by the system
        Return:
            scores (list): uniqueness scores for each recipe
        """
        ingredient_count = dict()
        scores = []
        
        for recipe in recipes:
            for ingredient in recipe.ingredients: 
                if ingredient.name in ingredient_count:
                    ingredient_count[ingredient.name] += 1
                else:
                    ingredient_count[ingredient.name] = 1
        
        for recipe in recipes:
            score = 0
            for ingredient in recipe.ingredients: 
                score += ingredient_count[ingredient.name]
            score = 1 / score 
            scores.append(score)

        return scores 
        
    def rank_recipes(self, recipes): 
        """
        Evaluates recipes based on two functions: recipe_uniqueness() and similarity_fitness(). 
        Higher score = higher quality recipe (unique and/or flavors are similar)
        Lower score = lower quality recipe (basic and/or flavors are not similar)
        Args:
            recipes (list): recipes generated by the system
        Return:
            recipe_ranks (list): a list of the evaluation scores for each recipe
        """
        fitness_scores = [recipe.similarity_fitness() for recipe in recipes]
        recipe_uniqueness = self.recipe_uniqueness(recipes)
        recipe_ranks = [fitness_scores[i] + recipe_uniqueness[i] for i in range(len(recipes))]
        return recipe_ranks 
 

def main():
    g = Generator()
    g.read_files("input/*.txt")

    recipes = g.generate(1, 0.8)

    recipe_ranks = g.rank_recipes(recipes)
    print(recipe_ranks)

    #for recipe in recipes:
       # recipe.name_recipe()
       # recipe.recipe_export(output_dir="output")
       # time.sleep(2) 

    max_index = recipe_ranks.index(max(recipe_ranks))
    best_recipe = recipes[max_index]
    print(best_recipe)
    

if __name__ == "__main__":
    main()
