"""
This modules implements the bulk of Bot Evolution.
"""
import pdb
import pickle
import os
import numpy as np
import copy
import settings
from utility import seq_is_equal, distance_between, angle_is_between, find_angle
from neural_network import NNetwork

np.random.seed()

RED = (255,0,0)
GREEN = (0,255,0)
colors = [RED, GREEN]
class Population:
    """
    The environment of bots and food.
    """

    def __init__(self, size, mutation_rate, no_food):
        assert(size >= 5)
        assert(0 < mutation_rate < 1)
        self.SIZE = size * 2
        self.mutation_rate = mutation_rate
        self.bots = []
        self.food = []
        self.time_since_last_death = 0.0

        # The neural network will have 1 neuron in the input layer, 1 hidden
        # layer with 2 neurons, and 4 neurons in the output layer. The "sigmoid"
        # activation function will be used on the hidden layer, and a "softmax"
        # activation function will be used on the output layer. Input consists
        # of the bot's direction and if there is or isn't food in the bots field
        # of vision. Output consists of whether or not to move foward, turn
        # left, turn right, or do nothing.
        for i in range(size):
            self.bots.append(Bot(NNetwork((2, 2, 4), ("sigmoid", "softmax")), RED, self))

        for i in range(size):
            self.bots.append(Bot(NNetwork((2, 2, 4), ("sigmoid", "softmax")), GREEN, self))
        for i in range(no_food):
            self.food.append(Food(self))

        # append strongest parents from previous run
        if os.path.isfile("modelc.pickle"):
            f = open("modelc.pickle")
            wts = pickle.load(f)
            f.close()
            bot = Bot(NNetwork((2, 2, 4), ("sigmoid", "softmax")), RED, self)
            bot.nnet.set_all_weights(wts)
            self.bots.append(bot)
            f = open("modelh.pickle")
            wts = pickle.load(f)
            f.close()
            bot = Bot(NNetwork((2, 2, 4), ("sigmoid", "softmax")), GREEN, self)
            bot.nnet.set_all_weights(wts)
            self.bots.append(bot)

    def eliminate(self, bot, replace = False):
        self.time_since_last_death = 0.0
        if bot in self.bots:
            self.bots.remove(bot)
        if replace:
            # random_rgb = (np.random.randint(30, 256), np.random.randint(30, 256), np.random.randint(30, 256))
            self.bots.append(Bot(NNetwork((2, 2, 4), ("sigmoid", "softmax")), bot.RGB, self))

    def feed(self, bot, food, is_bot = False):
        print "Feeding..."
        bot.score += 1.0
        if not is_bot:
            self.food.remove(food)
            self.food.append(Food(self))
        else:
            self.bots.remove(food)
        num_to_replace = int(self.SIZE / 7 - 1)
        if num_to_replace < 2:
            num_to_replace = 2
        for i in range(num_to_replace):
            weakest = self.bots[0]
            for other in self.bots:
                if other.score < weakest.score:
                    weakest = other
            self.eliminate(weakest)
        for i in range(num_to_replace):
            print "Replacing!!"
            if np.random.uniform(0, 1) <= self.mutation_rate:
                rgb = GREEN if is_bot else colors[np.random.randint(2)]
                # new_bot = Bot(bot.nnet, rgb, self)
                bot.x = np.random.randint(100, settings.WINDOW_WIDTH) * np.random.uniform(0, 1) #* settings.WINDOW_WIDTH / 2.0 + Bot.SPAWN_RADIUS # * np.random.uniform(0, 1) * np.cos(self.theta)
                bot.y = np.random.randint(100, settings.WINDOW_HEIGHT) * np.random.uniform(0, 1) #* settings.WINDOW_HEIGHT / 2.0 + Bot.SPAWN_RADIUS # * np.random.uniform(0, 1) * np.sin(self.theta)
                nb_c = bot.nnet.get_all_weights()
                mutated = False
                while not mutated:
                    # for k in range(len(nb_c)):
                    #     for i in range(len(nb_c[k])-1):
                    #         for j in range(len(nb_c[k][i])):
                    #             if np.random.uniform(0, 1) <= self.mutation_rate:
                    #                 nb_c[k][i][j] = nb_c[k][i][j] * np.random.normal(1, 0.5) + np.random.standard_normal()
                    #                 mutated = True
                    #                 print "Mutated!!"
                    # no of hidden units
                    for k in range(2):
                        if np.random.uniform(0, 1) <= self.mutation_rate:
                            idx1 = np.random.randint(2)
                            idx2 = np.random.randint(2)
                            nb_c[0][idx1][idx2] = nb_c[0][idx1][idx2] * np.random.normal(1, 0.5) + np.random.standard_normal()
                            mutated = True
                            bot.nnet.set_all_weights(nb_c)
                self.bots.append(bot)
            else:
                sorted_bots_by_score = sorted(self.bots, key=lambda x: x.score, reverse = True)
                # get first 2 strongest bots
                bot1, bot2 = sorted_bots_by_score[0], sorted_bots_by_score[1]
                conn1 = bot1.nnet.get_all_weights()
                conn2 = bot2.nnet.get_all_weights()
                conn3 = conn1
                idx1 = np.random.randint(2)
                idx2 = np.random.randint(2)
                if np.random.uniform(0, 1) <= self.mutation_rate:
                     conn3[0][idx1][idx2] = conn1[0][idx1][idx2]
                else:
                     conn3[0][idx1][idx2] = conn2[0][idx1][idx2]
                bot1.nnet.set_all_weights(conn3)
                self.bots.append(bot1)

        print "FED BOT!!!"

    def create_herbibots(self):
        for i in range(5):
            self.bots.append(Bot(NNetwork((2, 2, 4), ("sigmoid", "softmax")), GREEN, self))

    def create_carnibots(self):
        for i in range(5):
            self.bots.append(Bot(NNetwork((2, 2, 4), ("sigmoid", "softmax")), RED, self))

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

        for bot in self.bots[:]:
            if bot not in self.bots:
                continue

            sensory_input = []
            sensory_input_h = []
            chased = False
            chased_bot = None
                
            # This is where the bot's field of vision is put into action.
            min_theta = bot.theta - Bot.FIELD_OF_VISION_THETA / 2
            max_theta = bot.theta + Bot.FIELD_OF_VISION_THETA / 2
            food_in_sight = False
            
            # food in front
            if bot.RGB == GREEN:
                for food in self.food:
                    if angle_is_between(find_angle(bot.x, bot.y, food.x, food.y), min_theta, max_theta):
                        food_in_sight = True
                        break
                if food_in_sight:
                    sensory_input_h.append(1.0)
                else:
                    sensory_input_h.append(0.0)
                
                for bbot in self.bots:
                    if bbot.RGB == RED and angle_is_between(find_angle(bot.x, bot.y, bbot.x, bbot.y), min_theta, max_theta):
                        chased = True
                        chased_bot = bot
                        break
                if not chased:
                    sensory_input_h.append(0.0)
                elif chased:
                    sensory_input_h.append(1.0)

                
            # bot in front
            elif bot.RGB == RED:
                food_in_sight = False
                # version1 - check if bot in sight - eat that if it's there! :P
                idx = self.bots.index(bot)
                for bbot in self.bots:
                    if bbot.RGB == GREEN  and angle_is_between(find_angle(bot.x, bot.y, bbot.x, bbot.y), min_theta, max_theta):
                        food_in_sight = True
                        chased = True
                        chased_bot = bbot
                        break
                if food_in_sight:
                    # pdb.set_trace()
                    sensory_input.append(1.0)
                else:
                    sensory_input.append(0.0)

                sensory_input.append(0.0)

            if chased and bot.RGB == RED:
                for food in self.food:
                    if angle_is_between(find_angle(chased_bot.x, chased_bot.y, food.x, food.y), min_theta, max_theta):
                        food_in_sight = True
                        break
                if food_in_sight:
                    sensory_input_h = [1.0, 1.0]
                else:
                    sensory_input_h = [0.0, 1.0]

                chased_bot.update(dt, sensory_input_h)

            # Useful debugging outputs.
            #print(bot.RGB)
            #print(sensory_input)

            if bot.RGB == RED:
                bot.update(dt, sensory_input)
            elif bot.RGB == GREEN:
                bot.update(dt, sensory_input_h)

        if self.time_since_last_death >= 5:
            weakest = self.bots[0]
            for bot in self.bots:
                if bot.score < weakest.score:
                    weakest = bot
            self.eliminate(weakest, replace = True)

    def save_strongest_bots(self):
        sorted_bots_by_scorec = sorted((bot for bot in self.bots if bot.RGB == RED), key=lambda x: x.score, reverse = True)
        sorted_bots_by_scoreh = sorted((bot for bot in self.bots if bot.RGB == GREEN), key=lambda x: x.score, reverse = True)
        wts = sorted_bots_by_scorec[0].nnet.get_all_weights()
        f = open("modelc.pickle", "wb")
        pickle.dump(wts, f)
        f.close()
        wts = sorted_bots_by_scoreh[0].nnet.get_all_weights()
        f = open("modelh.pickle", "wb")
        pickle.dump(wts, f)
        f.close()

class Bot:
    """
    The representation of the circle thing with probes.
    """

    # In pixels/pixels per second/revolutions per second/radians.
    SPAWN_RADIUS = int(settings.WINDOW_WIDTH / 20) if settings.WINDOW_WIDTH <= settings.WINDOW_HEIGHT else int(settings.WINDOW_HEIGHT / 20)
    HITBOX_RADIUS = 6
    SPEED = 80.0
    SPEED_H = 0.9 * SPEED
    TURN_RATE = 2 * np.pi
    FIELD_OF_VISION_THETA = 45 * np.pi / 180

    # These lists represent the output from the neural network. Note that the
    # output '[0, 0, 0, 1]' means "do nothing".
    MOVE_FORWARD =  [1, 0, 0, 0]
    TURN_LEFT =     [0, 1, 0, 0]
    TURN_RIGHT =    [0, 0, 1, 0]
    MOVE_FASTER =    [0, 0, 0, 1]

    def __init__(self, nnet, rgb, population):
        self.nnet = nnet
        # self.nnet = copy.deepcopy(nnet)
        self.RGB = rgb
        self.pop = population
        self.theta = np.random.uniform(0, 1) * 2 * np.pi
        # self.x = settings.WINDOW_WIDTH / 2.0 + Bot.SPAWN_RADIUS * np.random.uniform(0, 1) * np.cos(self.theta)
        # self.y = settings.WINDOW_HEIGHT / 2.0 + Bot.SPAWN_RADIUS * np.random.uniform(0, 1) * np.sin(self.theta)
        self.x = np.random.randint(100, settings.WINDOW_WIDTH) * np.random.uniform(0, 1) #* settings.WINDOW_WIDTH / 2.0 + Bot.SPAWN_RADIUS # * np.random.uniform(0, 1) * np.cos(self.theta)
        self.y = np.random.randint(100, settings.WINDOW_HEIGHT) * np.random.uniform(0, 1) #* settings.WINDOW_HEIGHT / 2.0 + Bot.SPAWN_RADIUS # * np.random.uniform(0, 1) * np.sin(self.theta)
        
        self.score = 0.0

    def _move_forward(self, dt):
        speed = Bot.SPEED if self.RGB == RED else Bot.SPEED_H
        self.x += speed / settings.FPS * dt * np.cos(self.theta) * settings.TIME_MULTIPLIER
        self.y -= speed / settings.FPS * dt * np.sin(self.theta) * settings.TIME_MULTIPLIER
        if self.x < -Bot.HITBOX_RADIUS * 6 or self.x > settings.WINDOW_WIDTH + Bot.HITBOX_RADIUS * 6 \
        or self.y < -Bot.HITBOX_RADIUS * 6 or self.y > settings.WINDOW_HEIGHT + Bot.HITBOX_RADIUS * 6:
            self.pop.eliminate(self, replace = True)

    def _move_faster(self, dt):
        speed = Bot.SPEED * 1.1
        self.x += speed / settings.FPS * dt * np.cos(self.theta) * settings.TIME_MULTIPLIER
        self.y -= speed / settings.FPS * dt * np.sin(self.theta) * settings.TIME_MULTIPLIER
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

        no_herbibots = len(list(bot for bot in self.pop.bots if bot.RGB == GREEN))
        no_carnibots = len(list(bot for bot in self.pop.bots if bot.RGB == RED))
        if no_herbibots < 5:
            self.pop.create_herbibots()
        if no_carnibots < 5:
            self.pop.create_carnibots()
        
        # randomly mutate weights of bots!
            if np.random.uniform(0, 1) <= self.pop.mutation_rate:
                bot = self.pop.bots[np.random.randint(len(self.bots))]
                rgb = colors[np.random.randint(2)]
                # new_bot.x = np.random.randint(100, settings.WINDOW_WIDTH) * np.random.uniform(0, 1) #* settings.WINDOW_WIDTH / 2.0 + Bot.SPAWN_RADIUS # * np.random.uniform(0, 1) * np.cos(self.theta)
                # new_bot.y = np.random.randint(100, settings.WINDOW_HEIGHT) * np.random.uniform(0, 1) #* settings.WINDOW_HEIGHT / 2.0 + Bot.SPAWN_RADIUS # * np.random.uniform(0, 1) * np.sin(self.theta)
                nb_c = bot.nnet.get_all_weights()
                mutated = False
                k = np.random.randint(len(nb_c))
                i = np.random.randint(len(nb_c[k]-1))
                j = np.random.randint(len(nb_c[k][i]-1))
                nb_c[k][i][j] = nb_c[k][i][j] * np.random.normal(1, 0.5) + np.random.standard_normal()
                bot.nnet.set_all_weights(nb_c)
        

        if self.pop.time_since_last_death >= 0.2:
            for bot in self.pop.bots:
                if bot.RGB == GREEN and self.RGB == RED and distance_between(self.x, self.y, bot.x, bot.y) <= 2 * Bot.HITBOX_RADIUS :
                    print "Fed bot!"
                    self.pop.feed(self, bot, is_bot = True)
                    break

        # self.nnet.feed_forward(sensory_input)
        output = self.nnet.output(sensory_input)
        if seq_is_equal(output, Bot.MOVE_FORWARD):
            self._move_forward(dt)
        elif seq_is_equal(output, Bot.TURN_LEFT):
            self._turn_left(dt)
        elif seq_is_equal(output, Bot.TURN_RIGHT):
            self._turn_right(dt)
        elif seq_is_equal(output, Bot.MOVE_FASTER):
            if self.RGB == RED:
                self._move_forward(dt)
            else:
                self._move_faster(dt)


class Food:
    """
    The representation of the red circles.
    """

    # In pixels.
    HITBOX_RADIUS = 5
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
        for bot in self.pop.bots:
            if bot.RGB == GREEN and distance_between(self.x, self.y, bot.x, bot.y) <= Bot.HITBOX_RADIUS + Food.HITBOX_RADIUS:
                print "Fed bot!"
                self.pop.feed(bot, self)
                break
