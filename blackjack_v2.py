import gym
from gym import spaces
from gym.utils import seeding
import random
import math

def cmp(a, b):
    return float(a > b) - float(a < b)

# 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
NDECKS = 6
MIN_DECK_SIZE = 16



def usable_ace(hand):  # Does this hand have a usable ace?
    return 1 in hand and sum(hand) + 10 <= 21


def sum_hand(hand):  # Return current hand total
    if usable_ace(hand):
        return sum(hand) + 10
    return sum(hand)


def is_bust(hand):  # Is this hand a bust?
    return sum_hand(hand) > 21


def score(hand):  # What is the score of this hand (0 if bust)
    return 0 if is_bust(hand) else sum_hand(hand)


def is_natural(hand):  # Is this hand a natural blackjack?
    return sorted(hand) == [1, 10]

'''
Карты	Числовые значения
2	+0,5
3, 4	+1
5	+1,5
6	+1
7	+0,5
8	0
9	−0,5
10, В, Д, К, Т	−1
'''
halfs = [0, -1, 0.5, 1, 1, 1.5, 1, 0.7, 0, -0.5, -1]
def halfs_system(val):
    return halfs[val]
    
halfs_neg_probs = [0, 0.52, 1.10, 1.67, 2.20, 2.73, 3.20, 3.64, 4.01, 4.33, 4.63]
halfs_pos_probs = [0.09, 0.73, 1.36, 1.98, 2.64, 3.32, 4.07, 4.84, 5.65, 6.51, 7.39]
def halfs_probs(s):
    if s < 0:
        s = -s
        if s > 10:
            s = 10
        return halfs_neg_probs[s] * 0.01
    else:
        if s > 10:
            s = 10
        return halfs_pos_probs[s] * 0.01


class BlackjackEnvV2(gym.Env):
    """Simple blackjack environment

    Blackjack is a card game where the goal is to obtain cards that sum to as
    near as possible to 21 without going over.  They're playing against a fixed
    dealer.
    Face cards (Jack, Queen, King) have point value 10.
    Aces can either count as 11 or 1, and it's called 'usable' at 11.
    This game is placed with an infinite deck (or with replacement).
    The game starts with dealer having one face up and one face down card, while
    player having two face up cards. (Virtually for all Blackjack games today).

    The player can request additional cards (hit=1) until they decide to stop
    (stick=0) or exceed 21 (bust).

    After the player sticks, the dealer reveals their facedown card, and draws
    until their sum is 17 or greater.  If the dealer goes bust the player wins.

    If neither player nor dealer busts, the outcome (win, lose, draw) is
    decided by whose sum is closer to 21.  The reward for winning is +1,
    drawing is 0, and losing is -1.

    The observation of a 3-tuple of: the players current sum,
    the dealer's one showing card (1-10 where 1 is ace),
    and whether or not the player holds a usable ace (0 or 1).

    This environment corresponds to the version of the blackjack problem
    described in Example 5.1 in Reinforcement Learning: An Introduction
    by Sutton and Barto.
    http://incompleteideas.net/book/the-book-2nd.html
    """
    def __init__(self, natural=False):
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),
            spaces.Discrete(11),
            spaces.Discrete(2),
            spaces.Discrete(41)
            ))
        self.seed()
        
        self.ndecks = NDECKS * 4
        self.count_decks = 0
        self.shuffle_decks()
        self.deck_status = 0

        # Flag to payout 1.5 on a "natural" blackjack win, like casino rules
        # Ref: http://www.bicyclecards.com/how-to-play/blackjack/
        self.natural = natural
        # Start the first game
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def shuffle_decks(self):
        self.decks = deck * self.ndecks
        self.count_decks = 0
        random.shuffle(self.decks)
        
    def draw_card(self):
        if len(self.decks) <= MIN_DECK_SIZE:
            self.shuffle_decks()
        res = self.decks.pop()
        self.count_decks += halfs_system(res)
        self.deck_status = int(self.count_decks / math.ceil(len(self.decks) / 52)) + 20
        '''
        if self.deck_status < 0:
            self.deck_status = 0
        if self.deck_status > 20:
            self.deck_status = 20
        '''

        return res


    def draw_hand(self):
        return [self.draw_card(), self.draw_card()]


    def step(self, action):
        assert self.action_space.contains(action), 'action = '.format(action)
        if action == 1:  # hit: add a card to players hand and return
            self.player.append(self.draw_card())
            if is_bust(self.player):
                done = True
                reward = -1.
            else:
                done = False
                reward = 0.
        elif action == 2: #double
            self.player.append(self.draw_card())
            done = True
            while sum_hand(self.dealer) < 17:
                self.dealer.append(self.draw_card())
            
            reward = cmp(score(self.player), score(self.dealer))
            if self.natural and is_natural(self.player) and reward == 1.:
                reward = 1.5 * 2
            else:
                reward = reward * 2
            
        else:  # stick: play out the dealers hand, and score
            done = True
            while sum_hand(self.dealer) < 17:
                self.dealer.append(self.draw_card())
            reward = cmp(score(self.player), score(self.dealer))
            if self.natural and is_natural(self.player) and reward == 1.:
                reward = 1.5
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return (sum_hand(self.player), self.dealer[0], usable_ace(self.player), self.deck_status)

    def reset(self):
        self.dealer = self.draw_hand()
        self.player = self.draw_hand()
        return self._get_obs()
    
    def get_count_decks(self):
        s = self.count_decks / math.ceil(len(self.decks) / 52)
        return halfs_probs(int(s))
    
    
def main():
    env = BlackjackEnvV2()
    print(env.reset())
    print(int(len(env.decks) / 52))
    r = env.get_count_decks()
    print(r)
    
    
    
if __name__ == "__main__":
    main()
    
# -*- coding: utf-8 -*-

