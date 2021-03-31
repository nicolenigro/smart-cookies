"""
PQ2: Smart Cookies
CSCI 3725
Team OOG: Ahmed Hameed, Adrienne Miller, Nicole Nigro
Version Date: 3/30/21

Description: This system uses a modified genetic algorithm to generate, output, and evaluate cookie recipes. 

Dependencies: glob, json, numpy, os, random, requests, string, time, bs4
"""

import json, os, random, string, time, requests
import numpy as np

import glob
from bs4 import BeautifulSoup

from flavor_pairing import similarity, pairing

ESSENTIAL_INGREDIENTS = ["all-purpose flour", "sugar", "butter", "flour", "egg", "eggs", "yolk", "yolks", "baking soda", \
    "baking powder", "salt", "vanilla", "vegetable oil"]
RECIPE_NAMES = dict()

class Ingredient:
    def __init__(self, name, quantity, unit="grams", essential=False):
        """
        Initializes an ingredient with a name, quantity, unit, and whether or not it's essential.
        Args:
            name (str): name of the ingredient
            quantity (float): amount of the ingredient
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


    def __repr__(self):
        """
        Method that returns a representation of an Ingredient object
        Args: 
            None
        Return: 
            output (str): The string representing Ingredient object
        """
        return f"Ingredient({self.name}, {self.quantity}, {self.unit}, {self.essential})"


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
        self.score = 0
        self.typicality = 0 


    def similarity_fitness(self):
        """
        Function that calculates the 'similarity fitness' score of a recipe. Iterates through all 
        non-essential ingredients and uses flavor_pairing.py to retrieve the similarity of that
        non-essential ingredient with every other non-esential ingredient. If there is no match for
        an ingredient in the npy file, 0.5 is automatically added -- if an ingredient is not in the
        npy Ingredient list, it is from our input recipes, and we can assume that it is a relatively
        normal ingredient. 
        Args:
            None
        Return:
            fitness_score (int): the fitness score of a recipe
        """
        fitness_score = 0
        non_essentials = self.get_non_essentials() # gathering all non-essential or mix-in ingredients

        # for every non-essential/mix-in ingredient 
        for i in range(len(non_essentials)):
            ingred1 = non_essentials[i].name
            # compare the current mix-in ingredient to every other mix-in ingredient

            for j in range(i, len(non_essentials)):
                ingred2 = non_essentials[j].name
                try:
                    sim = similarity(ingred1, ingred2)
                except:
                    # ingredient isn't in numpy flavor profile dataset but it still comes from cookie inspiring
                    # set so we still know that the ingredient will pair well with other cookie mix-ins
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
        Deletes an Ingredient object from the Recipe.
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
        Args:
            None
        Return:
            name (str): Recipe name
        """
        non_essentials = self.get_non_essentials()
        # if there are no non-essential ingredients --> recipe is basic
        if len(non_essentials) == 0:
            name_ingredient = "Basic"
        else:
            # pick a random non essential ingredient
            name_ingredient = np.random.choice(non_essentials).name

        # get list of words and associated weights based on occurrenc
        related_words, weights = self.name_recipe_helper(name_ingredient)
        if len(related_words) == 0: # website does not have any matches 
            adjective = "Simple"
        else:
            # select adjective based on weighted scores from website
            adjective = ''.join(random.choices(related_words, weights, k=1))
        name = f"{string.capwords(adjective)} {string.capwords(name_ingredient)} Cookies"

        # if the generated name is already being used for a different recipe then we update the occurrence count for 
        # that name and add the new count to the end of the generated name (ex. Simple Sugar Cookies 3)
        if name in RECIPE_NAMES:
            RECIPE_NAMES[name] += 1
            name += " " + str(RECIPE_NAMES[name])
        # otherwise, make a new entry and count in the dictionary for the recipe name
        else:
            RECIPE_NAMES[name] = 1

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
        with open(output_path, "w", encoding='utf-8') as f:

            # write each ingredient and quantity to line 
            for ingredient in self.ingredients:
                line = f"{round(ingredient.quantity,2)} grams {ingredient.name} \n"
                f.write(line)
            f.write("\n")

            # write instructions to file
            f.write("Instructions")
            f.write(self.instructions)


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
            output += (str)(item.quantity) + " grams " + item.name + "\n"

        output += "\nInstructions"
        output += self.instructions
        return output


    def __repr__(self):
        """
        Method that returns a representation of a Recipe object
        Args: 
            None
        Return: 
            output (str): The string representing Recipe object
        """
        return f"Recipe({self.name}, {self.ingredients}, {self.instructions})"


class Generator:
    def __init__(self, recipe_type):
        """
        Initializes a Generator Object with an empty ingredient_names dictionary, an empty recipes list, 
        and a counter for new recipes (initalized at 1).
        Args: 
            recipe_type (str): type of recipe to be created
        Return:
            None
        """
        self.recipe_type = recipe_type # used in __str__ method
        self.ingredient_names = dict()  #all ingredients in all recipes
        self.recipes = [] # all recipes in current population
        self.current_generation = 1 # used in __str__ method
        self.new_recipe_count = 0
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
            
            with open(os.path.join(filename), encoding='utf-8') as f:
                input_string = ""
                for line in f.readlines():  # add each ingredient line in file to recipe
                    if ("grams" in line):
                        # split line into list of form [quantity, ingredient]
                        split_line = line.rstrip().split(" grams ")
                        ingredient_name = split_line[1]
                        ingredient_quantity = round(float(split_line[0]), 2)
                        ingredient = Ingredient(name=ingredient_name, quantity=ingredient_quantity)
                        ingredients.append(ingredient)
                    input_string += line

                    # update dictionary mapping ingredient occurence to counts
                    if ingredient_name not in self.ingredient_names:
                        self.ingredient_names[ingredient_name] = 1
                    else: 
                        self.ingredient_names[ingredient_name] += 1
                
                split_input = input_string.split("Instructions")
                instructions = split_input[1]

                # get rid of 'input' in filename
                filename = filename.split('/')[1]

                # create recipe with all ingredients in a file
                recipe = Recipe(name=filename, ingredients=ingredients, instructions=instructions)

                # save recipe to the list of recipes
                self.recipes.append(recipe)


    def crossover(self):
        """
        Weighted by similarity_fitness, selects two 'parent' recipes to be crossed over. Then, 
        selects a random pivot index in each parent's nonessentials and generates a new child recipe. 
        Args: 
            None 
        Return: 
            child (Recipe): new Recipe object generated by crossing over two parents in the current population
        """
        # calculate fitness of each recipe
        recipes_fitness = [recipe.similarity_fitness() for recipe in self.recipes]
        parent1, parent2 = random.choices(population=self.recipes, weights=recipes_fitness, k=2)

        # extract parent1 essentials and nonessentials
        non_essentials_parent1 = []
        essentials_parent1 = []
        for ingredient in parent1.ingredients:
            if ingredient.essential:
                essentials_parent1.append(ingredient)
            else:
                non_essentials_parent1.append(ingredient)

        # extract parent2 nonessentials
        non_essentials_parent2 = []
        for ingredient in parent2.ingredients:
            if not ingredient.essential:
                non_essentials_parent2.append(ingredient)

        # select random pivots for nonessential ingredients
        pivot_1 = random.randint(0, len(non_essentials_parent1))
        pivot_2 = random.randint(0, len(non_essentials_parent2))

        # create subset of parent recipes that will be used in child
        subgroup_recipe1 = non_essentials_parent1[1:pivot_1]
        subgroup_recipe2 = non_essentials_parent2[pivot_2:]

        # create child recipe object
        child_name = "recipe" + str(self.new_recipe_count) + ".txt"

        # always use parent1 essentials
        child_ingredients = essentials_parent1 + subgroup_recipe1 + subgroup_recipe2 

        # always use parent1 instructions
        child = Recipe(name=child_name, ingredients=child_ingredients, instructions=parent1.instructions)
        child.combine_duplicates()

        self.new_recipe_count += 1

        return child


    def mutation(self, recipe, prob_of_mutation):
        """
        Mutates a recipe by randomly selecting a mutation type (adding or deleting an ingredient)
        and then executing that mutation.
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

        # extract nonessential ingredients
        for ingredient in recipe.ingredients:
            if not ingredient.essential:
                non_essentials.append(ingredient)

        if (current_mutation == "add"):
            # randomly ordered list of nonessentials
            random_order = random.choices(non_essentials, k=len(non_essentials))
            found_ingredient = False

            # loop through nonessential ingredients until you find one that is in npy file
            for i in range(len(random_order)):
                reference_ingredient = random_order[i]
                try:
                    # an ingredient needs to have a flavor pairing of 0.5 or higher to be considered a good pairing
                    good_pairings = pairing(reference_ingredient.name, 0.5)
                    good_ingredients, weights = good_pairings.items()
                    new_ingredient = random.choices(good_ingredients, weights=weights)
                except: 
                    continue # ingredient wasn't in numpy file so we continue to the next ingredient
                found_ingredient = True
                break
            
            # we didn't find a nonessential in npy file -- pick a random one
            if not found_ingredient:
                new_ingredient = random.choice(self.default_mix_ins)

            # vanilla is a very common cookie recipe ingredient and serves as an excellent base for
            # gathering all cookie ingredients in a particular category 
            all_spices = pairing("vanilla", 0.00, cat="spice")
            all_herbs = pairing("vanilla", 0.00, cat="herb")
            
            # if the ingredient is a spice or herb, then it should be used in a smaller quantity (1-20 grams)
            if new_ingredient in all_spices or new_ingredient in all_herbs:
                new_ingredient_amount = random.uniform(1, 20)
            # otherwise, the ingredients isn't a spice or herb so we use normal quantities (10-100 grams)
            else:
                new_ingredient_amount = random.uniform(10, 100)
                
            recipe.add_ingredient(Ingredient(new_ingredient, round(new_ingredient_amount, 2)))
            # end of 'add' ingredient mutation type

        elif current_mutation == "delete":
            # select random ingredient to remove from recipe
            ingredient_to_delete = random.choice(recipe.ingredients)
            recipe.delete_ingredient(ingredient_to_delete)

        recipe.combine_duplicates() # combine any duplicate ingredients in recipe and add their amounts
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
        Performs the genetic algorithm (GA) for num_generations and returns the final recipes generated. 
        Args: 
            num_generations (int): number of new generations to be created before returning final recipe generation
            mutation_prob (float): the probability of mutation
        Return: 
            self.recipes (Recipe[]): final list of recipes resulting from several generations of crossover 
        """
        for i in range(num_generations):
            self.generate_population(mutation_prob)
            self.current_generation += 1

        # the recipes returned are the most fit recipes after running the GA for num_generations
        return self.recipes


    def get_default_list(self): 
        """
        Populates our default_mix_ins list with spices, herbs, and nuts.
        Args:
            None
        Return:
            None
        """ 
        # extract relavent categories of ingredients from npy file
        all_spices = pairing("vanilla", 0.00, cat="spice")
        all_herbs = pairing("vanilla", 0.00, cat="herb")
        all_nuts = pairing("vanilla", 0.00, cat="nut/seed/pulse")

        # add to default mix in lists
        self.default_mix_ins += list(all_spices.keys())
        self.default_mix_ins += list(all_herbs.keys())
        self.default_mix_ins += list(all_nuts.keys())
    

    def recipe_novelty(self, recipes): 
        """ 
        Iterate over the recipes our system generated, keeping track of how often ingredients appear.
        A higher novelty scores means more unique ingredients; a lower novelty score means mostly
        generic ingredients.
        Args:
            recipes (list): recipes generated by the system
        Return:
            scores (list): novelty scores for each recipe
        """
        ingredient_count = dict()
        scores = []
        
        # create dictionary mapping ingredient names to the number of times they appear in recipes
        for recipe in recipes:
            for ingredient in recipe.ingredients: 
                if ingredient.name in ingredient_count:
                    ingredient_count[ingredient.name] += 1
                else:
                    ingredient_count[ingredient.name] = 1
        
        # calculate novelty score for each recipe 
        for recipe in recipes:
            score = 0
            for ingredient in recipe.ingredients: 
                score += 1 / ingredient_count[ingredient.name]
            scores.append(score)

        return scores 
    
    
    def recipe_typicality(self, recipes):
        """
        Returns the 'typicality' value of a specific recipe. Similar to the recipe_novelty() function, 
        except compares recipes to the ingredient counts of the inspiring set, instead of in the 
        current generation. Higher typicality score means that a recipe is more similar to those in the 
        inspiring set. 
        Args:
            recipes (list): recipes generated by the system
        Return: 
            scores (list): list of typicality scores 
        """
        for recipe in recipes:
            score = 0
            num_ingredients = 0 
            for ingredient in recipe.ingredients:
                # if it's not in inspiring set --> not typical and neccesary for comparison
                if ingredient.name in self.ingredient_names: 
                    score += self.ingredient_names[ingredient.name]
                    num_ingredients += 1
            avg_score = score / num_ingredients
            recipe.typicality = avg_score

    
    def calculate_ranks(self, recipes): 
        """
        Evaluates recipes based on two functions: recipe_novelty() and similarity_fitness(). 
        Higher score = higher quality recipe (unique and/or flavors are similar)
        Lower score = lower quality recipe (basic and/or flavors are not similar)
        Args:
            recipes (list): recipes generated by the system
        Return:
            None
        """
        fitness_scores = [recipe.similarity_fitness() for recipe in recipes]
        recipe_novelty = self.recipe_novelty(recipes)
        index = 0
        for recipe in recipes:
            recipe.score = fitness_scores[index] + recipe_novelty[index]
            index += 1
    

    def export_rankings(self, recipes):
        """
        Export the rankings to a file in the metrics folder. Each file lists the average combined
        novelty and quality score and average typicality score at the top, followed by the ranked recipes
        and each recipe's novelty/quality and typicality scores.
        Args:
            recipes (list): recipes generated by the system
        Return:
            None
        """
        self.calculate_ranks(recipes)
        self.recipe_typicality(recipes)
        ranked_recipes = sorted(recipes, key=lambda Recipe: Recipe.score, reverse=True)
        
        path = "metrics/" + "metrics" + ".txt"
        with open(path, "w", encoding='utf-8') as f:

            avg_quality = np.mean([recipe.score for recipe in recipes])
            avg_typicality = np.mean([recipe.typicality for recipe in recipes])
            f.write(f"Average Combined Novelty and Quality Score: {str(round(avg_quality, 2))}, Average Typicality Score: {str(round(avg_typicality, 2))} \n \n")
            
            counter = 1
            for recipe in ranked_recipes:
                formatted_recipe = f"{str(counter)}. {recipe.name}, score: {str(round(recipe.score, 2))}, typicality: {str(round(recipe.typicality, 2))} \n"
                f.write(formatted_recipe)
                counter += 1


    def __str__(self): 
        """
        Method for printing out a Generator object. 
        Args:
            None
        Return:
            output (str): The string output representing a Generator object
        """ 
        return f"{self.recipe_type} Recipe Generator on Generation {self.current_generation}"
    

    def __repr__(self):
        """
        Method that returns a representation of a Generator object
        Args: 
            None
        Return: 
            output (str): The string representing Generator object
        """
        return f"Generator({self.recipe_type})"      



def main():
    g = Generator("Cookie")
    g.read_files("input/*.txt")

    recipes = g.generate(5, 0.2)
    for recipe in recipes:
       recipe.name_recipe()
       recipe.recipe_export(output_dir="output")
       time.sleep(1) 

    g.export_rankings(recipes)

if __name__ == "__main__":
    main()