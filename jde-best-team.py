import argparse
import csv
import json
import random
import collections
import numpy
from objectpath import *
from deap import creator, base, tools, algorithms

INPUT_FILE = 'result.csv'
BUDGET = 90000000
FORMATIONS = [['P', 'A', 'A', 'I', 'I']]
NGEN = 15 # nb of generations
MU = 200 # size of pop
LAMBDA = 1000 # size of children
CXPB = 0.7 # prop of crossover
MUTPB = 0.3 # prop of mutation

# manage args
parser = argparse.ArgumentParser(description='Compute best team for JDE')
parser.add_argument('-c', '--config',nargs='?', default='config.csv', \
                    help='config file')
args = parser.parse_args()

try:
    with open(args.config) as jsonfile:
        conf = json.load(jsonfile)
        INPUT_FILE = conf.get('INPUT_FILE', INPUT_FILE)
        BUDGET = conf.get('BUDGET', BUDGET)
        FORMATIONS = conf.get('FORMATIONS', FORMATIONS)
        NGEN = conf.get('NGEN', NGEN)
        MU = conf.get('MU', MU)
        LAMBDA = conf.get('LAMBDA', LAMBDA)
        CXPB = conf.get('CXPB', CXPB)
        MUTPB = conf.get('MUTPB', MUTPB)
except FileNotFoundError:
    exit('File Not Found: ' + args.config)

# create a dedicated fitness obj
creator.create("TeamFitness", base.Fitness, weights=(1.0,))

#### CLASSES
class PickError(Exception):
    pass

class Team:
    def __init__(self, price_limit, formations, generator):
        # init fitness
        self.fitness = creator.TeamFitness()

        self.dead = False

        self.price_limit = price_limit
        self.remaining_budget = price_limit

        # randomly choose a formation
        self.formation = random.choice(formations)

        self.players = {}

        # pick a new team
        t = generator.pickTeam(self.formation, self.remaining_budget, [])
        self.dead = t['dead']
        self.remaining_budget = t['remaining_budget']
        self.players = {x: self.players.get(x, []) + t['new_players'].get(x, []) for x in set(self.players).union(t['new_players'])}

        # manage dead team
        generator.isDead(self)

    # compute price
    def totalPrice(self):
        return sum([sum(player['current_price'] for player in pos) for k, pos in self.players.items()])

    # compute score
    def totalScore(self):
        return sum([sum(player['score'] for player in pos) for k, pos in self.players.items()])

    # print team
    def printMe(self):
        print('#######')
        if(self.dead == True):
            print('/!\/!\/!\ DEAD /!\/!\/!\ ')
        print('- formation -')
        print(self.formation)
        for k, pos in self.players.items():
            print('-- ' + k + ' --')
            for player in pos:
                print(player)
        print('- score -')
        print(self.totalScore())
        print('- price -')
        print(self.totalPrice())
        print('#######')

class TeamTools:
    def __init__(self, data):
        self.data = data

    # pick one new player in data
    def pickOne(self, position, remaining_budget, current_team):
        # manage max player per team = 3 and avoid player duplicate
        teams = {}
        full_team = -1
        players = []
        for p in current_team:
            if(p['team_id'] in teams):
                teams[p['team_id']] += 1
            else:
                teams[p['team_id']] = 1
            if(teams[p['team_id']] == 3):
                full_team = p['team_id']
            players.append(p['player_name'])

        try:
            if(full_team != -1):
                picked = random.choice([i for i in self.data[position] if i['current_price'] < remaining_budget and i['player_name'] not in players and i['team_id'] != full_team])
            else:
                picked = random.choice([i for i in self.data[position] if i['current_price'] < remaining_budget and i['player_name'] not in players])
            return picked
        except IndexError: # try to choice in an empty list
            raise PickError("IndexError")
        except KeyError: # try to choice for a missing position
            raise PickError("KeyError")

    # pick a new team to complete the cur_team with a specified formation and budget
    def pickTeam(self, formation, remaining_budget, cur_team):
        pos = formation.copy()
        random.shuffle(pos)
        res = {'dead': False, 'remaining_budget': remaining_budget, 'new_players': {}}
        while(len(pos) > 0):
            cur_pos = pos.pop()
            try:
                p = self.pickOne(cur_pos, remaining_budget, cur_team)
            except PickError:
                res['dead'] = True
                continue

            res['remaining_budget'] -= p['current_price']
            cur_team.append(p)
            if(cur_pos not in res['new_players']):
                res['new_players'][cur_pos] = []
            res['new_players'][cur_pos].append(p)
        return res

    # return if team is dead and refresh dead status
    def isDead(self, team):
        # already dead ?
        if(team.dead == True):
            return True
        # a team with more than 3 players
        teams = {}
        for t in [p['team_id'] for k, pos in team.players.items() for p in pos]:
            if(t in teams):
                teams[t] += 1
            else:
                teams[t] = 1
            if(teams[t] > 3):
                team.dead = True
                return True
        # same player
        players = []
        for n in [p['player_name'] for k, pos in team.players.items() for p in pos]:
            if(n in players):
                team.dead = True
                return True
            players.append(n)
        # price over limit ?
        if(team.totalPrice() > team.price_limit):
            team.dead = True
            return True

        return False

    # evaluation func
    def evalTeam(self, team):
        # if team is dead under evaluate it
        if(self.isDead(team)):
            return 0,
        else:
            weight = team.totalScore()
            return weight,

    # cross over func
    def cxTeams(self, team1, team2):
        # generate pool of players to pick in
        data = {x: team1.players.get(x, []) + team2.players.get(x, []) for x in set(team1.players).union(team2.players)}
        # create 2 teams based on this pool
        tt = TeamTools(data)
        n_team1 = Team(BUDGET, [team1.formation, team2.formation], tt)
        n_team2 = Team(BUDGET, [team1.formation, team2.formation], tt)
        return n_team1, n_team2

    # mutation func
    def mutTeam(self, team):
        if(self.isDead(team)):
            return team,

        # 2 types of mutations : formation and players permutation

        tt = TeamTools(team.players)
        n_team = Team(BUDGET, FORMATIONS, tt)

        # if formation changed then no need to permute player
        if(team.formation != n_team.formation):
            n_team.dead = False
            # retrieve missing position in new team
            missing = []
            target = collections.Counter(n_team.formation)
            cur_team = []
            for p in list(target.keys()):
                to_be_added = 0
                if(p not in n_team.players):
                    to_be_added = target[p]
                else:
                    to_be_added = target[p] - len(n_team.players[p])
                    cur_team += n_team.players[p]
                missing = missing + [p] * to_be_added

            # retrieve player to fill these missing positions
            t = self.pickTeam(missing, n_team.remaining_budget, cur_team)

            n_team.dead = t['dead']
            n_team.remaining_budget = t['remaining_budget']
            n_team.players = {x: n_team.players.get(x, []) + t['new_players'].get(x, []) for x in set(n_team.players).union(t['new_players'])}

            if(self.isDead(n_team)):
                return team,
            else:
                return n_team,
        else: # if same formation then permute player
            if(self.isDead(n_team)):
                return team,
            # pick random position
            pos = random.choice(list(n_team.players.keys()))
            # pick random index
            ind = random.randrange(len(n_team.players[pos]))
            # define max price of the new player
            price = n_team.players[pos][ind]['current_price'] + n_team.remaining_budget
            # pick new player
            new_p = random.choice([i for i in self.data[pos] if i['current_price'] < price])
            # update player
            n_team.players[pos][ind] = new_p
            return n_team,

#### LOAD DATA
data = {}
with open(INPUT_FILE, newline='') as csvfile:
    players = csv.DictReader(csvfile)
    for player in players:
        if(player['POSITION'] not in data):
            data[player['POSITION']] = []
        data[player['POSITION']].append({'player_name': player['PLAYER_NAME'], 'team_id': player['TEAM_ID'], 'current_price': int(player['CURRENT_PRICE']), 'score': float(player['SCORE'])})

print('total players imported : ' + str(sum(len(data[p]) for p in list(data.keys()))))

#### ALGO GEN
toolbox = base.Toolbox()

team_tools = TeamTools(data)
toolbox.register("individual", Team, BUDGET, FORMATIONS, team_tools)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", team_tools.evalTeam)
toolbox.register("mate", team_tools.cxTeams)
toolbox.register("mutate", team_tools.mutTeam)
toolbox.register("select", tools.selNSGA2)

#### exec
pop = toolbox.population(n=MU)
hof = tools.ParetoFront()
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", numpy.mean, axis=0)
stats.register("std", numpy.std, axis=0)
stats.register("min", numpy.min, axis=0)
stats.register("max", numpy.max, axis=0)

algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats, halloffame=hof)

#### print best team
print('HOF')
print(hof[0].printMe())