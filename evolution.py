import copy
import operator

import numpy as np

from player import Player


class Evolution:
    def __init__(self):
        self.game_mode = "Neuroevolution"

    def next_population_selection(self, players, num_players):
        """
        Gets list of previous and current players (μ + λ) and returns num_players number of players based on their
        fitness value.

        :param players: list of players in the previous generation
        :param num_players: number of players that we return
        """
        # TODO (Implement top-k algorithm here)
        # players = players.sort(key=lambda x: x.fitness, reverse=True)


        # TODO (Additional: Implement roulette wheel here)
        # players = self.roulette_wheel(players, num_players)

        # TODO (Additional: Implement SUS here)
        # players = self.SUS(players, num_players)

        # Q_tournament (apparently works better)
        players = self.Q_tournament(players, num_players, 4)

        players.sort(key=lambda x: x.fitness, reverse=True)

        # TODO (Additional: Learning curve)
        best_fitness = players[0].fitness
        worst_fitness = players[len(players) - 1].fitness
        fitnesses = [player.fitness for player in players]
        mean_fitness = sum(fitnesses) / len(fitnesses)
        f = open("result.txt", 'a')
        f.write(str(best_fitness) + " " + str(mean_fitness) + " " + str(worst_fitness) + "\n")

        return players[: num_players]

    def roulette_wheel(self, players, num_players):
        next_gen = []
        probabilities = []
        total_fitness = sum([player.fitness for player in players])
        for player in players:
            probabilities.append(player.fitness/total_fitness)
        # probabilities = [player.fitness/total_fitness for player in players]
        for i in range(num_players):
            choice = np.random.choice(players, p=probabilities)
            next_gen.append(choice)
        return next_gen

    def SUS(self, players, num_players):
        N2 = num_players
        step = 1 / N2
        # length = 1 - (1 / N2)
        next_gen = []
        probabilities = []
        total_fitness = sum([player.fitness for player in players])
        for player in players:
            probabilities.append(player.fitness/total_fitness)
            # 8
            # 1/8 -> 0.1
            # 0.4 0.2 0.2 0.2

        cumulative_prob = []
        j = 0
        for i in range(0, len(probabilities)):
            j += probabilities[i]
            cumulative_prob.append(j)

        start = np.random.uniform(0, step, 1) # 0.1

        for i in range(num_players):
            point = start + (step * i)
            for j, p in enumerate(cumulative_prob):
                if point <= p:
                    choice = players[j]
                    next_gen.append(choice)
                    break

        return next_gen

    def Q_tournament(self, players, num_players, q):
        next_gen = []
        for i in range(num_players):
            candidates = np.random.choice(players, q)
            next_gen.append(max(candidates, key=operator.attrgetter('fitness')))
        return next_gen

    def generate_new_population(self, num_players, prev_players=None):
        """
        Gets survivors and returns a list containing num_players number of children.

        :param num_players: Length of returning list
        :param prev_players: List of survivors
        :return: A list of children
        """
        first_generation = prev_players is None
        if first_generation:
            return [Player(self.game_mode) for _ in range(num_players)]
        else:
            # TODO ( Parent selection and child generation )
            # new_players = prev_players  # DELETE THIS AFTER YOUR IMPLEMENTATION
            # print("prev parents", len(prev_players))

            # choose parents
            prev = prev_players.copy()
            # chosen_parents = self.roulette_wheel(prev, num_players)
            # chosen_parents = self.SUS(prev, num_players)
            chosen_parents = self.Q_tournament(prev, num_players, 4)

            new_players = self.crossover_operation(chosen_parents)
            for child in new_players:
                self.mutation(child)

            return new_players

    def crossover_operation(self, prev_players):
        children = []
        for i in range(0, len(prev_players), 2):
            parent1 = prev_players[i]
            parent2 = prev_players[i+1]

            child1 = self.clone_player(parent1)
            child2 = self.clone_player(parent2)

            x = np.random.uniform(0.0, 1.0)


            # do crossover
            if x < 0.8:
                parent1_w1 = parent1.nn.W[0]
                parent1_b1 = parent1.nn.b[0]
                parent1_w2 = parent1.nn.W[1]
                # parent1_b2 = parent1.nn.b[1]

                parent2_w1 = parent2.nn.W[0]
                parent2_b1 = parent2.nn.b[0]
                parent2_w2 = parent2.nn.W[1]
                # parent2_b2 = parent2.nn.b[1]

                split_point = np.random.randint(1, len(parent1_w1)-1)

                for i, row in enumerate(parent1_w1):
                    p1_first_part = row[0:split_point]
                    p2_second_part = parent2_w1[i][split_point:int(len(row))]
                    p2_first_part = parent2_w1[i][0:split_point]
                    p1_second_part = row[split_point:int(len(row))]
                    child1.nn.W[0][i] = np.append(p1_first_part, p2_second_part)
                    child2.nn.W[0][i] = np.append(p2_first_part, p1_second_part)

                for i, row in enumerate(parent1_w2):
                    p1_first_part = row[0:split_point]
                    p2_second_part = parent2_w2[i][split_point:int(len(row))]
                    p2_first_part = parent2_w2[i][0:split_point]
                    p1_second_part = row[split_point:int(len(row))]
                    child1.nn.W[1][i] = np.append(p1_first_part, p2_second_part)
                    child2.nn.W[1][i] = np.append(p2_first_part, p1_second_part)

                child1.nn.b[0][0:split_point] = parent1_b1[0:split_point]
                child1.nn.b[0][split_point:len(parent1_b1)] = parent2_b1[split_point:len(parent1_b1)]

                child2.nn.b[0][0:split_point] = parent2_b1[0:split_point]
                child2.nn.b[0][split_point:len(parent1_b1)] = parent1_b1[split_point:len(parent1_b1)]

            children.append(child1)
            children.append(child2)

        return children

    def mutation(self, child):
        pm = 0.3
        mu, sigma = 0.2, 0.5
        for i in range(len(child.nn.W)):
            x = np.random.uniform(0.0, 1.0)
            if x < pm:
                child.nn.W[i] += np.random.randn(child.nn.W[i].shape[0] * child.nn.W[i].shape[1]).reshape(child.nn.W[i].shape[0], child.nn.W[i].shape[1])

        for i in range(len(child.nn.b)):
            x = np.random.uniform(0.0, 1.0)
            if x < pm:
                child.nn.b[i] += np.random.normal(mu, sigma, child.nn.b[i].shape)

    def clone_player(self, player):
        """
        Gets a player as an input and produces a clone of that player.
        """
        new_player = Player(self.game_mode)
        new_player.nn = copy.deepcopy(player.nn)
        new_player.fitness = player.fitness
        return new_player
