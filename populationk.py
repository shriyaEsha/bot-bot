"""
This modules implements the bulk of Bot Evolution.
"""

import numpy as np
import copy
import settings
from utility import seq_is_equal, distance_between, angle_is_between, find_angle
from neural_network import NeuralNet
import h5py

class Population:
    """
    The environment of bots and food.
    """

    def __init__(self, size, mutation_rate, no_food):
        assert(size >= 5)
        assert(0 < mutation_rate < 1)
        self.SIZE = size
        self.mutation_rate = mutation_rate
        self.bots = []
        self.food = []
        self.time_since_last_death = 0.0

        # The neural network will have 1 neuron in the input layer, 1 hidden
        # layer with 2 neurons, and 4 neurons in the output layer. The sigmoid
        # activation function will be used on the hidden layer, and a softmax
        # activation function will be used on the output layer. Input consists
        # of the bot's direction and if there is or isn't food in the bots field
        # of vision. Output consists of whether or not to move foward, turn
        # left, turn right, or do nothing.
        neural_net_example = NeuralNet((2, 5, 4), ("sigmoid", "softmax"), "c")
        colors = [(255,0,0), (0,255,0)]
        # spawning equal no of carni_bots and herbi_bots
        for i in range(size):
            random_rgb = colors[0]
            example_bot = Bot(neural_net_example, random_rgb, self)
            self.bots.append(example_bot)

        # neural_net_example = NeuralNet((2, 5, 4), ("sigmoid", "softmax"), "h")
        for i in range(size):
            random_rgb = colors[1]
            example_bot = Bot(neural_net_example.change_type("h"), random_rgb, self)
            self.bots.append(example_bot)
        for i in range(no_food):
            self.food.append(Food(self))

    def eliminate(self, bot, replace = False):
        self.time_since_last_death = 0.0
        self.bots.remove(bot)
        colors = [(255,0,0), (0,255,0)]
        neural_net_example = NeuralNet((2, 5, 4), ("sigmoid", "softmax"), "c")

        if replace:
            random_rgb = colors[np.random.randint(0, 2)]
            t = "c" if random_rgb == (255,0,0) else "h"
            example_bot = Bot(neural_net_example.change_type(t), random_rgb, self)
            self.bots.append(example_bot)

    def feed(self, bot, food, is_bot=False):
        bot.score = 1.0
        if is_bot == False:
            # # print "Bot-Food collision"
            self.food.remove(food)
            self.food.append(Food(self))
        else: # bot-bot collision
            # # print "Bot-Bot collision"
            idx = self.bots.index(bot)
            self.bots.pop(idx)
            if len(self.bots) <= 7:
                # # print "Creating new Bots!"
                neural_net_example = NeuralNet((2, 5, 4), ("sigmoid", "softmax"), "c")
                colors = [(255,0,0), (0,255,0)]
                for i in range(self.SIZE):
                    random_rgb = colors[np.random.randint(0, 2)]
                    t = "c" if random_rgb == (255,0,0) else "h"
                    # neural_net_example = NeuralNet((2, 5, 4), ("sigmoid", "softmax"), t)
                    example_bot = Bot(neural_net_example.change_type(t), random_rgb, self)
                    self.bots.append(example_bot)

        num_to_replace = int(self.SIZE / 7 - 1)
        if num_to_replace < 2:
            num_to_replace = 2
        for i in range(num_to_replace):
            weakest = self.bots[0]
            for other in self.bots:
                if other.score < weakest.score:
                    weakest = other
            self.eliminate(weakest)
        colors = [(255,0,0), (0,255,0)]
        # asexual reproduction - mutation
        for i in range(num_to_replace):
            if np.random.uniform(0, 1) <= self.mutation_rate:
                new_rgb = colors[np.random.randint(0,2)]
                new_bot = Bot(bot.nnet, new_rgb, self)
                # new_bot.x = bot.x + Bot.HITBOX_RADIUS * 4 * np.random.uniform(0, 1) * np.random.choice((-1, 1))
                # new_bot.y = bot.y + Bot.HITBOX_RADIUS * 4 * np.random.uniform(0, 1) * np.random.choice((-1, 1))
                new_bot.x = np.random.randint(settings.WINDOW_WIDTH) + Bot.HITBOX_RADIUS * 4 * np.random.uniform(0, 1) * np.random.choice((-1, 1))
                new_bot.y = np.random.randint(settings.WINDOW_HEIGHT) + Bot.HITBOX_RADIUS * 4 * np.random.uniform(0, 1) * np.random.choice((-1, 1))
                
                # nn
                nb_c = new_bot.nnet.get_all_weights()
                mutated = False
                while not mutated:
                    for k in range(len(nb_c)-1):
                        # # print "len: ",len(nb_c)
                        # # print "from: ",len(nb_c[k])
                        # # print "to: ",len(nb_c[k][0]-1)
                        for i in range(len(nb_c[k])-1):
                            for j in range(len(nb_c[k][0])):
                                if np.random.uniform(0, 1) <= self.mutation_rate:
                                    nb_c[k][i][j] = nb_c[k][i][j] * np.random.normal(1, 0.5) + np.random.standard_normal()
                                    mutated = True
                    # for k in range(len(nb_c)):
                    #     if np.random.uniform(0,1) <= self.mutation_rate:
                    #         i = np.random.randint(0, len(nb_c[k]))
                    #         j = np.random.randint(0, len(nb_c[k][0]))
                    #         nb_c[k][i][j] = nb_c[k][i][j] * np.random.normal(1, 0.5) + np.random.standard_normal()
                    #         mutated = True
                new_bot.nnet.set_all_weights(nb_c)
                self.bots.append(new_bot)
                self.bots[self.bots.index(new_bot)].RGB = (255,255,255)

            # sexual - crossover
            else:
                # # print "Sexual! :P"
                sorted_bots_by_score = sorted(self.bots, key=lambda x: x.score, reverse = True)
                # get first 2 strongest bots
                bot1, bot2 = sorted_bots_by_score[0], sorted_bots_by_score[1]
                bot1.change_color((255,255,255))  
                conn1 = bot1.nnet.get_all_weights()
                conn2 = bot2.nnet.get_all_weights()
                # get random weight to crossover
                idx1 = np.random.randint(len(conn1))
                idx2 = np.random.randint(len(conn1[idx1]))
                idx3 = np.random.randint(len(conn1[idx1][idx2]))
                conn3 = conn1
                conn3[idx1][idx2][idx3] = conn2[idx1][idx2][idx3]
                new_bot = Bot(bot.nnet, bot.RGB, self)
                new_bot.nnet.set_all_weights(conn3)
                new_bot.x = bot.x + Bot.HITBOX_RADIUS * 4 * np.random.uniform(0, 1) * np.random.choice((-1, 1))
                new_bot.y = bot.y + Bot.HITBOX_RADIUS * 4 * np.random.uniform(0, 1) * np.random.choice((-1, 1))

                self.bots.append(new_bot)

    def update(self, dt):
        """
        Updates the population's internals. The bulk of event handling for all
        bots and food starts here.
        """
        self.time_since_last_death += 1.0 / settings.FPS * dt * settings.TIME_MULTIPLIER

        for food in self.food[:]:
            if food not in self.food:
                continue
            food.update(dt)
        colors = [(255,0,0), (0,255,0)]
        for bot in self.bots[:]:
            if bot.RGB == (255,255,255) and self.time_since_last_death >= 0.26:
                idx = self.bots.index(bot)
                self.bots[idx].RGB = colors[np.random.randint(2)]
            if bot not in self.bots:
                continue

            sensory_input = []

            # This is where the bot's field of vision is put into action.
            min_theta = bot.theta - Bot.FIELD_OF_VISION_THETA / 2
            max_theta = bot.theta + Bot.FIELD_OF_VISION_THETA / 2
            food_in_sight = False
            # food in front
            for food in self.food:
                if angle_is_between(find_angle(bot.x, bot.y, food.x, food.y), min_theta, max_theta):
                    food_in_sight = True
                    break
            if food_in_sight:
                sensory_input.append(1.0)
            else:
                sensory_input.append(0.0)
            
            # bot in front
            food_in_sight = False
            # version1 - check if bot in sight - eat that if it's there! :P
            idx = self.bots.index(bot)
            for bbot in self.bots:
                if bot.RGB != bbot.RGB and self.bots.index(bbot) != idx and angle_is_between(find_angle(bot.x, bot.y, bbot.x, bbot.y), min_theta, max_theta):
                    food_in_sight = True
                    break
            if food_in_sight:
                sensory_input.append(1.0)
            else:
                sensory_input.append(0.0)

            # Useful debugging outputs.
            # # # print(bot.RGB)
            # # # print(sensory_input)

            bot.update(dt, sensory_input)

        if self.time_since_last_death >= 5:
            weakest = self.bots[0]
            for bot in self.bots:
                if bot.score < weakest.score:
                    weakest = bot
            self.eliminate(weakest, replace = True)

    def save_strongest_bots(self):
        sorted_bots_by_scorec = sorted((bot for bot in self.bots if bot.RGB == (255,0,0)), key=lambda x: x.score, reverse = True)
        sorted_bots_by_scoreh = sorted((bot for bot in self.bots if bot.RGB == (0,255,0)), key=lambda x: x.score, reverse = True)
        sorted_bots_by_scorec[0].nnet.model.save("modelc.h5")
        sorted_bots_by_scoreh[0].nnet.model.save("modelh.h5")


class Bot:
    """
    The representation of the circle thing with probes.
    """

    # In pixels/pixels per second/revolutions per second/radians.
    SPAWN_RADIUS = int(settings.WINDOW_WIDTH / 20) if settings.WINDOW_WIDTH <= settings.WINDOW_HEIGHT else int(settings.WINDOW_HEIGHT / 20)
    HITBOX_RADIUS = 5
    SPEED = 50.0
    TURN_RATE = 2 * np.pi
    FIELD_OF_VISION_THETA = 45 * np.pi / 180

    # These lists represent the output from the neural network. Note that the
    # output '[0, 0, 0, 1]' means "do nothing".
    MOVE_FORWARD =  [1, 0, 0, 0]
    TURN_LEFT =     [0, 1, 0, 0]
    TURN_RIGHT =    [0, 0, 1, 0]

    def __init__(self, nnet, rgb, population):
        self.nnet = copy.deepcopy(nnet)
        # print "NNet: ",self.nnet, "rgb: ",rgb
        self.RGB = rgb
        self.pop = population
        self.theta = np.random.uniform(0, 1) * 2 * np.pi
        # self.x = settings.WINDOW_WIDTH / 2.0 + Bot.SPAWN_RADIUS * np.random.uniform(0, 1) * np.cos(self.theta)
        # self.y = settings.WINDOW_HEIGHT / 2.0 + Bot.SPAWN_RADIUS * np.random.uniform(0, 1) * np.sin(self.theta)
        self.x = np.random.randint(100, settings.WINDOW_WIDTH) * np.random.uniform(0, 1) #* settings.WINDOW_WIDTH / 2.0 + Bot.SPAWN_RADIUS # * np.random.uniform(0, 1) * np.cos(self.theta)
        self.y = np.random.randint(100, settings.WINDOW_HEIGHT) * np.random.uniform(0, 1) #* settings.WINDOW_HEIGHT / 2.0 + Bot.SPAWN_RADIUS # * np.random.uniform(0, 1) * np.sin(self.theta)
        
        self.score = 0.0

    def change_color(self, color):
        self.RGB = color

    def _move_forward(self, dt):
        self.x += Bot.SPEED / settings.FPS * dt * np.cos(self.theta) * settings.TIME_MULTIPLIER
        self.y -= Bot.SPEED / settings.FPS * dt * np.sin(self.theta) * settings.TIME_MULTIPLIER
        if self.x < -Bot.HITBOX_RADIUS * 6 or self.x > settings.WINDOW_WIDTH + Bot.HITBOX_RADIUS * 6 \
        or self.y < -Bot.HITBOX_RADIUS * 6 or self.y > settings.WINDOW_HEIGHT + Bot.HITBOX_RADIUS * 6:
            self.pop.eliminate(self, replace = True)

    def _turn_left(self, dt):
        self.theta += Bot.TURN_RATE / settings.FPS * dt * settings.TIME_MULTIPLIER
        while self.theta >= 2 * np.pi:
            self.theta -= 2 * np.pi

    def _turn_right(self, dt):
        self.theta -= Bot.TURN_RATE / settings.FPS * dt * settings.TIME_MULTIPLIER
        while self.theta < 0:
            self.theta += 2 * np.pi

    def update(self, dt, sensory_input):
        """
        Updates the bot's internals. "Hunger" can be thought of as a score
        between '-1' and '1' where a greater value means less hungry.
        """
        self.score -= 1.0 / settings.FPS / 10.0 * dt * settings.TIME_MULTIPLIER
        if self.score < -1:
            self.score = -1.0

        """
        Updates the bot's internals and handles bot<->bot collision.
        """
        # make them eat a certain period after spawning
        # # # print "pop time: ",self.pop.time_since_last_death
        if self.pop.time_since_last_death >= 0.3:
            idx = self.pop.bots.index(self)
            for bbot in self.pop.bots:
                if self.RGB != bbot.RGB and self.pop.bots.index(bbot) != idx and distance_between(self.x, self.y, bbot.x, bbot.y) <= Bot.HITBOX_RADIUS + Bot.HITBOX_RADIUS:
                    herbi_bot = bbot if bbot.RGB == (0,255,0) else self
                    self.pop.feed(bbot, self, True)
                    break

        # neural network - feedforward
        # self.nnet.feed_forward(sensory_input)
        # print "Model before output: ",self.nnet
        output = self.nnet.output(sensory_input)
        if seq_is_equal(output, Bot.MOVE_FORWARD):
            self._move_forward(dt)
        elif seq_is_equal(output, Bot.TURN_LEFT):
            self._turn_left(dt)
        elif seq_is_equal(output, Bot.TURN_RIGHT):
            self._turn_right(dt)

class Food:
    """
    The representation of the red circles.
    """

    # In pixels.
    HITBOX_RADIUS = 10
    RGB = (255, 255, 0)

    def __init__(self, population):
        mid_x = int(settings.WINDOW_WIDTH / 2)
        mid_y = int(settings.WINDOW_HEIGHT / 2)
        max_left_x = mid_x - (Bot.SPAWN_RADIUS + Bot.HITBOX_RADIUS + 5)
        min_right_x = mid_x + (Bot.SPAWN_RADIUS + Bot.HITBOX_RADIUS + 5)
        max_top_y = mid_y - (Bot.SPAWN_RADIUS + Bot.HITBOX_RADIUS + 5)
        min_bottom_y = mid_y + (Bot.SPAWN_RADIUS + Bot.HITBOX_RADIUS + 5)
        self.x = np.random.choice((np.random.uniform(0, max_left_x), np.random.uniform(min_right_x, settings.WINDOW_WIDTH)))
        self.y = np.random.choice((np.random.uniform(0, max_top_y), np.random.uniform(min_bottom_y, settings.WINDOW_HEIGHT)))
        self.pop = population

    def update(self, dt):
        """
        Updates the food's internals and handles bot<->food collision.
        """
        # for herbi_bots
        for bot in self.pop.bots:
            if bot.RGB == (0,255,0) and distance_between(self.x, self.y, bot.x, bot.y) <= Bot.HITBOX_RADIUS + Food.HITBOX_RADIUS:
                self.pop.feed(bot, self)
                break
