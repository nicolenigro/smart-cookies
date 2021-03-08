"""
PQ2: Smart Cookies
CSCI 3725
Team OOG: Ahmed Hameed, Adrienne Miller, Nicole Nigro
Version Date: 3/8/21

Description: This system generates cookie recipes.

Dependencies: glob, numpy, os, random
"""

"""
Week of March 8th --> 
- naming function in Recipe class
- updating readings in files [X]
- mutations: incorporate flavor pairings
    - mutations for base ingredients: 
- crossover --> only do on non essentials 
- update functions in recipe (add_ingredient, remove_ingredient)
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
        -Special naming function
            *pick from list of adjectives, pick one of the mix-ins, "cookies"
                *alliteration name
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
"""

import glob
import numpy as np
import os
import random
import ingredient_and_recipe as food

WORD_EMBED_VALS = np.load('ingred_word_emb.npy', allow_pickle=True).item()
INGRED_CATEGORIES = np.load('ingred_categories.npy', allow_pickle=True).item()
INGREDIENT_LIST = sorted(WORD_EMBED_VALS.keys())
ESSENTIAL_INGREDIENTS = ["sugar", "butter", "flour", "eggs", "baking soda", "baking powder", "salt"]

class Ingredient:
    def __init__(self, name, quantity, unit="grams", essential=False):
        """
        Initializes an ingredient with a name and a quantity.
        Args:
            name (str): name of the ingredient
            unit (str): unit of measurement
            quantity (float): amount of the ingredient, measured in oz
            essential (bool): if it is an essential ingredient or not
        """
        self.name = name
        self.unit = unit
        self.quantity = abs(quantity)
        self.essential = essential

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
        
    def set_name(self, new_name):
        """
        Sets the name of ingredient to new_name.
        Args:
            new_name (str): the new name for the ingredient
        Return:
            None
        """
        self.name = new_name
    
    def get_name(self):
        """
        Gets the name of the current ingredient.
        Args: 
            None
        Return:
            self.name (str): the name of the ingredient
        """
        return self.name    

    def set_quantity(self, new_quantity):
        """
        Sets the quantity of an ingredient.
        Args:
            new_quantity (float): the new quantity of the ingredient
        Return:
            None
        """
        self.quantity = new_quantity
        
    def get_quantity(self):
        """
        Returns the quantity of the current ingredient.
        Args:
            None
        Return:
            self.quantity (float): the quantity of the ingredient
        """
        return self.quantity
    
    def similarity(self, n2):
        """Returns the similarity between two ingredients based on our data."""
        v1 = WORD_EMBED_VALS[self.name]
        v2 = WORD_EMBED_VALS[n2]
        return np.dot(v1, v2)

    def pairing(self, threshold, cat=None):
        """
        Describes what flavors are similar to the specified ingredient based on
        a similarity threshold and an optional category to which the flavors
        belong.
        """
        ingr = self.name
        pairings = {}
        ilist = list(set(INGREDIENT_LIST) - set([ingr]))
        for i in ilist:
            if similarity(ingr, i) >= threshold:
                if cat is not None:
                    if i in INGRED_CATEGORIES:
                        if INGRED_CATEGORIES[i] == cat:
                            pairings[i] = similarity(ingr, i)
                else:
                    pairings[i] = similarity(ingr, i)
        for key, value in sorted(pairings.items(), key=lambda kv: (kv[1],kv[0]), \
        reverse=True):
            print(key, value)

    def request_pairing(self, threshold, cat=None):
        """Displays a specific pairing to the user in a readable way."""
        ingr = self.name
        if cat:
            print("\nWhat pairs well with " + ingr + " that is a " + cat + "?")
            pairing(ingr, threshold, cat)
        else:
            print("\nWhat pairs well with " + ingr + "?")
            pairing(ingr, threshold)


class Recipe:
    def __init__(self, name, ingredients, instructions):
        """
        Initializes a recipe with a name and a list of Ingredients that make up the recipe.
        Args:
            name (str): The name of the recipe (make creative later on) 
            ingredients (list): List of Ingredient objects
            instructions (str): 
        """
        self.name = name
        self.ingredients = ingredients
        self.instructions = instructions
        self.base_ingredients = []
        self.non_essential_ingredients = []
        self.sort_ingredients()
    
    def sort_ingredients(self): 
        """
        Goes through ingredients list and marks which ingredients are essential and which ones are mix-ins.
        Args:
            None
        Return:
            None 
        """
        for ingredient in self.ingredients: 
            ingredient.mark_essential_ingredient()
            if ingredient.essential: 
                self.base_ingredients.append(ingredient)
            else:
                self.non_essential_ingredients.append(ingredient)

    def normalize(self, normal_value):
        """
        Normalizes the ingredient quantities in recipe to add up to normal_value. If quantities are not adding
        up to the given normal value, changes the ingredient amounts with a calculation.
        Args:
            normal_value (int) - The value that ingredient amounts will be normalized to. Units may vary.
        Return:
            None
        """
        total = 0
        
        #calculate the total oz that the recipe currently sums up to
        for item in self.ingredients:
            amount = item.get_quantity()
            total += amount
        
        if (total == normal_value):
            return
        else: 
            for i in range(len(self.ingredients)):
                item = self.ingredients[i]
                amount = item.get_quantity()
                # normalization occurs by taking proportion of quantity:total and multiplying it by a
                # normalization value
                normalized = (amount/total)*normal_value
                item.set_quantity(normalized)
                self.ingredients[i] = item
    
    def fitness(self):
        """
        Calculates the fitness of the current recipe by reading through the list of ingredients and
        counting the number of unique ingredients present in that recipe. Essentially, more 
        ingredients = higher fitness score.
        Args:
            None
        Return:
            fitness_score (int): the fitness score of a recipe
        """
        fitness_score = 0
        
        previously_seen_ingredients = [] #list of ingredient names (str)
        for item in self.ingredients:
            ing_name = item.get_name
            if (ing_name in previously_seen_ingredients):
                pass
            else:
                previously_seen_ingredients.append(ing_name)
                fitness_score += 1
        
        return fitness_score

    def reset_ingredient_amount(self, index):
        """
        Resets the ingredient amount to a randomly selected value between 0.5 and 100. 
        Args:
            index (int): The index of the ingredient that whose quantity we want to reset
        Return:
            None
        """
        new_amount = random.uniform(1, 50)
        self.ingredients[index].set_quantity(new_amount)

    def set_ingredient_name(self, index, new_name):
        """
        Sets the recipe ingredient at the given index to new_name.
        Args:
            index (int): The index of the ingredient whose name we want to overwrite
            new_name (str): The new name of ingredient
        Return:
            None
        """
        self.ingredients[index].set_name(new_name)

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
    
    def name_recipe(self):
        """
        Name a recipe with an adjective, ingredient name, cookies
        """

        # select ingredient randomly from nonessential ingredient list 

        non_essentials = []
        for ingredient in self.ingredients: 
            if not ingredient.essential: 
                non_essentials.append(ingredient)
                
        name_ingredient = np.random.choice(non_essentials)
        
        #url = f"https://describingwords.io/for/{name_ingredient}
        # name = adjective + ingredient + cookies
    
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
            output += item.get_name() + " " + (str)(item.get_quantity()) + "\n"
        
        output += "\nInstructions"
        output += self.instructions
        return output
        

class Generator: 
    def __init__(self):
        """
        Initializes an Generator Object with an empty ingredient_names list (all ingredients in all recipes), 
        an empty recipe list (all recipes in current population) and a counter for new recipes (initalized at 1)
        Args: 
            None
        """
        self.ingredient_names = [] #all ingredients in population
        self.recipes = []
        self.new_recipe_count = 1

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
        for filename in glob.glob(input_directory): # open each example recipe file
            ingredients = [] # intialize list of all ingredients in current recipe 
            f = open(os.path.join(filename))
            
            input_string = ""
            for line in f.readlines(): # add each ingredient line in file to recipe
                if ("grams" in line):
                    split_line = line.rstrip().split(" grams ") # split line into list of form [quantity, ingredient]
                    ingredient_name = split_line[1]
                    ingredient_quantity = round(float(split_line[0]), 2)
                    ingredient = food.Ingredient(name=ingredient_name,quantity=ingredient_quantity)
                    ingredients.append(ingredient)
                input_string += line

                # add to list of all ingredients in example file set
                if ingredient_name not in self.ingredient_names:
                    self.ingredient_names.append(ingredient_name)

            split_input = input_string.split("Instructions")
            instructions = split_input[1]

            recipe = food.Recipe(name=filename, ingredients=ingredients, instructions=instructions) #create recipe with all ingredients in a file

            # combine duplicate ingredients and normalize recipe to 100 oz
            #recipe.combine_duplicates()
            #recipe.normalize()

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
        recipes_fitness = [recipe.fitness() for recipe in self.recipes]
        parent1, parent2 = random.choices(population=self.recipes, weights=recipes_fitness, k=2)
    
        # make sure that recipes always have at least 2 ingredients
        pivot_1 = random.randint(1, parent1.fitness())
        pivot_2 = random.randint(0, parent2.fitness()-1)

        # create subset of parent recipes that will be used in child
        first_subgroup_recipe1 = parent1.ingredients[0:pivot_1]
        second_subgroup_recipe2 = parent2.ingredients[pivot_2:]

        # create child recipe object
        child_name = "recipe" + str(self.new_recipe_count) + ".txt"
        child_ingredients = first_subgroup_recipe1 + second_subgroup_recipe2
        child = Recipe(name=child_name, ingredients=child_ingredients)
        child.combine_duplicates()
        child.normalize()
        
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
        mutation_types = ["change_amount", "switch_ingredients", "add", "delete"]
        current_mutation = random.choice(mutation_types)
        
        if (current_mutation == "change_amount"):
            # select random index of ingredient to change amount of
            current_index = random.randint(0, len(recipe.ingredients)-1)
            recipe.reset_ingredient_amount(current_index)
        elif (current_mutation == "switch_ingredients"):
            # select random index to change ingredient and random ingredient to change to
            current_index = random.randint(0, len(recipe.ingredients)-1)
            new_name = random.choice(self.ingredient_names)
            recipe.set_ingredient_name(current_index, new_name)
        elif (current_mutation == "add"):
            # select random ingredient and quantity to add to recipe 
            new_ingredient_name = random.choice(self.ingredient_names)
            new_ingredient_amount = random.uniform(1, 50)
            recipe.add_ingredient(Ingredient(new_ingredient_name, new_ingredient_amount))
        elif (current_mutation == "delete"):
            # select random ingredient to remove from recipe 
            ingredient_to_delete = random.choice(recipe.ingredients)
            recipe.delete_ingredient(ingredient_to_delete)
        
        recipe.combine_duplicates()
        recipe.normalize()
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
            mutated_recipe = self.mutation(recipe=new_recipe, prob_of_mutation=mutation_prob)
            mutated_recipe.normalize()
            next_generation.append(mutated_recipe)

        # sort previous and next generation by their fitness
        sorted_previous = sorted(prev_generation, key=lambda recipe: recipe.fitness(), reverse=True)
        sorted_next = sorted(next_generation, key=lambda recipe: recipe.fitness(), reverse=True) 
        
        # take top 50% of each generation to create the new recipe 'population'
        midpoint_index = len(sorted_previous) // 2
        new_population = sorted_previous[0:midpoint_index] + sorted_next[0:midpoint_index]

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


def main():
    g = Generator()
    g.read_files("input/*.txt")
    recipes = g.recipes

    for recipe in recipes: 
        print()
        print(recipe)

if __name__ == "__main__":
    main()