import math
import random
import logging
import numpy as np
import itertools
from abc import ABC, abstractmethod

from phases import Phases
from cards import Card, Sorcery, Creature, Land


# from player import Player


class Game:
    def __init__(self, players):
        self.players = players
        for index, player in enumerate(self.players):
            player.index = index

        self.starting_hand_size = 7
        self.attackers = []
        self.blockers = []
        self.battlefield = []
        self.stack_is_empty = True
        self.temporary_zone = []
        self.damage_targets = []
        self.active_player = self.players[random.randint(0, len(self.players) - 1)]
        self.nonactive_player = self.players[1 - self.active_player.index]
        self.player_just_moved = self.active_player
        self.player_with_priority = self.active_player
        self.current_phase_index = Phases.BEGINNING_PHASE
        # two counters to help keep track of damage assignment - per attacker - per blocker, respectively
        self.attacker_counter = 0
        self.blocker_counter = 0
        self.current_attacker # reference to current attacker for blocker order unroller
        

    def get_board_string(self, colors=["Gold", "Silver"], additional_attacker_names=None, additional_block_assignments=None, additional_order_assignments=None, additional_mana_info=None):
        current_player = self.player_with_priority
        other_player = self.players[1 - current_player.index]

        # append additional attacker info 
        # used with AttackerUnrollers for returning updated state information
        attacker_names = [attacker.name_id for attacker in self.attackers]
        if additional_attacker_names != None:
            for attacker_name in additional_attacker_names:
                if attacker_name not in attacker_names:
                    attacker_names.append(attacker_name)

        attacker_block_dict = {}
        # build blocking dictionary
        # loop over attackers
        for attacker in self.attackers:
            attacker_block_dict[attacker.name_id] = []
            # loop over blocking info stored in game state
            for blocker in attacker.is_blocked_by:
                attacker_block_dict[attacker.name_id].append(blocker.name_id)
            # add additional passed info
            if attaker.name_id in additional_block_assignments:
                for blocker_name in additional_block_assignments[attacker.name_id]:
                    if blocker_name not in attacker_block_dict[attacker.name_id]:
                        attacker_block_dict[attacker.name_id].append(blocker_name)
            if len(attacker_block_dict[attacker.name_id]) == 0:
                attacker_block_dict[attacker.name_id].append("None")

        # convert attacker_block_dict to str
        
        attacker_block_str = []
        for attacker_name, blocker_names in attacker_block_dict.items():
            if len(blocker_names == 0):
                blocker_names = ["None"]
            attacker_str = f'attacker: ${attacker_name}$ blocked by: ${'$'.join(blocker_names)}$'
            attacker_block_str.append(attacker_str)
        attacker_block_str = '\n'.join(attacker_block_str)
        if len(attacker_block_str) == 0:
            attacker_block_str = "None"

        #attacker_string = '$'.join(attacker_names)

        # get land info
        lands = self.get_lands()
        lands_current = self.filter_cards_player(lands, current_player)
        lands_other = self.filter_cards_player(lands, other_player)
        lands_current_untapped = self.filter_cards_tapped(lands_current, tapped=False)
        lands_current_tapped = self.filter_cards_tapped(lands_current, tapped=True)
        lands_other_untapped = self.filter_cards_tapped(lands_other, tapped=False)
        lands_other_tapped = self.filter_cards_tapped(lands_other, tapped=True)

        # get general creature info
        creatures = self.get_creatures()
        creatures_current = self.filter_cards_player(creatures, current_player)
        creatures_other = self.filter_cards_player(creatures, other_player)
        creatures_current_untapped = self.filter_cards_tapped(creatures_current, tapped=False)
        creatures_current_tapped = self.filter_cards_tapped(creatures_current, tapped=True)
        creatures_other_untapped = self.filter_cards_tapped(creatures_other, tapped=False)
        creatures_other_tapped = self.filter_cards_tapped(creatures_other, tapped=True)


        # get blocking order assignment info if blocking order phase
        # This info may not be necessary
        damage_order_str = "None"
        if additional_order_assignments not None:
            attacker_name = additional_order_assignments["attacker_name"]
            blocker_names = additional_order_assignments["blocker_names"]
            damage_order_str = f'attacker: ${attacker_name}$ blocked by: ${'$'.join(blocker_names)'

        # get manapool and debt info
        if additional_mana_info is None:
            manapool = current_player.manapool
            debt = current_player.generic_debt
        else:
            manapool = additional_mana_info['manapool']
            debt = additional_mana_info['debt']

        # convert manapool to str
        manapool_str = '$ '.join([f'{mana}$ {value}' for (mana,value) in manapool.items()])


        # remember, this could be broken down into to different compoments
        # "missing" parts of game state suck as summoning sickness, graveyards, etc


        board_string = f'''
        player-color$ {colors[current_player.index]}$
        life$ {current_player.life}$
        opponent-life$ {other_player.life}$
        phase$ {self.current_phase_index}$
        hand$ {self.cards_to_string(current_player.hand)}$
        generic-debt$ {debt}$
        manapool$ {manapool_str}$
        opponent-cards$ {len(other_player.hand)}$
        self-lands-untapped$ {self.cards_to_string(lands_current_untapped)}$
        self-lands-tapped$ {self.cards_to_string(lands_current_tapped)}$
        opponent-lands-untapped$ {self.cards_to_string(lands_other_untapped)}$
        opponent-lands-tapped$ {self.cards_to_string(lands_other_tapped)}$
        self-creatures-untapped$ {self.cards_to_string(creatures_current_untapped)}$
        self-creatures-tapped$ {self.cards_to_string(creatures_current_tapped)}$
        opponent-creatures-untapped$ {self.cards_to_string(creatures_other_untapped)}$
        opponent-creatures-tapped$ {self.cards_to_string(creatures_other_tapped)}$
        attackers-blockers$ {attacker_block_str}$
        damage-order$ {damage_order_str}$
        '''
    # maybe add _id
    def cards_to_string(self,cards):
        return '$'.join(cards.name)

    def get_lands(self):
        lands = []
        for i in range(len(self.battlefield)):
            permanent = self.battlefield[i]
            if isinstance(permanent, Land):
                lands.append(permanent)
        return lands

    def get_creatures(self):
        creatures = []
        for i in range(len(self.battlefield)):
            permanent = self.battlefield[i]
            if isinstance(permanent, Creature):
                creatures.append(permanent)
        return creatures
    
    def filter_cards_player(self, cards, player):
        new_cards = []
        for card in cards:
            if card.owner == player:
                new_cards.append(land)
        return new_cards

    def filter_cards_tapped(self, cards, tapped=False):
        new_cards = []
        for card in cards:
            if card.is_tapped == tapped:
                new_cards.append(card)
        return new_cards


    def update_damage_targets(self):
        self.damage_targets = []
        self.damage_targets = self.get_battlefield_creatures() + self.players

    def damage_target_strings(self):
        strings = []
        creatures = self.get_battlefield_creatures()
        for creature in creatures:
            strings.append(str(creature))
        strings.extend(self.player_target_to_string())
        return strings

    def get_moves(self):
        player = self.player_with_priority
        move_indexes, move_strings = self.get_legal_moves(player)
        return move_indexes

    def get_results(self, player_index):
        player = self.players[player_index]
        opponent = self.players[1 - player.index]
        assert self.is_over()
        if player.has_lost and opponent.has_lost:
            return 0.5
        if player.has_lost:
            return 0.0
        if opponent.has_lost:
            return 1.0

    def make_move(self, move, verbose=False, attackers_passed=False, blockers_passed=False, assignments_passed=False):
        player = self.player_with_priority
        self.player_just_moved = player
        if player.generic_debt > 0:
            for mana in move:
                player.manapool[mana] -= 1
                player.generic_debt -= 1
            return True
        if player.casting_spell != "":
            if player.casting_spell == "Vengeance":
                dead_creature = self.battlefield[move]
                self.battlefield.remove(dead_creature)
                dead_creature.owner.graveyard.append(dead_creature)
            if player.casting_spell == "Stone Rain":
                destroyed_land = self.battlefield[move]
                self.battlefield.remove(destroyed_land)
                destroyed_land.owner.graveyard.append(destroyed_land)
            if player.casting_spell == "Index":
                for i in range(len(move)):
                    indexed_card = player.deck.pop()
                    indexed_card.deck_location_known = True
                    self.temporary_zone.append(indexed_card)
                # TODO: Consider if the logic behind declaring blockers, declaring attackers and assigning combat damage
                #       can be simplified in a similar manner by allowing moves to be a list of lists
                for index in move:
                    player.deck.append(self.temporary_zone[index])
                self.temporary_zone = []
            if player.casting_spell == "Lava Axe":
                self.players[move].life -= 5
            if player.casting_spell == "Rampant Growth":
                if not move == "Refuse":
                    land_index = player.find_land_in_library(move)
                    land = player.deck.pop(land_index)
                    self.battlefield.append(land)
                    land.is_tapped = False
                    land.owner = player
                player.shuffle_deck()
            if player.casting_spell == "Volcanic Hammer":
                self.update_damage_targets()
                self.damage_targets[move].take_damage(3)
            if player.casting_spell == "Sacred Nectar":
                player.life += 4
            player.casting_spell = ""
            return True

        if move is "Pass":
            player.passed_priority = True
            self.player_with_priority = self.active_player.get_opponent(self)
            if self.players[0].passed_priority and self.players[1].passed_priority and self.stack_is_empty:
                self.go_to_next_phase()
            return True
        if self.current_phase_index == Phases.MAIN_PHASE_PRE_COMBAT:
            playable_indices = player.get_playable_cards(self)
            callable_permanents, ability_indices = player.get_activated_abilities(self)
            if move < len(playable_indices):
                player.play_card(playable_indices[move], self, verbose)
            else:
                move -= len(playable_indices)
                for i in range(len(ability_indices)):
                    if move > ability_indices[i]:
                        move -= ability_indices[i]
                    else:
                        callable_permanents[i].use_tapped_ability(move - 1)

        if self.current_phase_index == Phases.DECLARE_ATTACKERS_STEP:
            attacking_player = self.active_player
            attacking_player.has_attacked = True
            eligible_attackers = attacking_player.get_eligible_attackers(self)

            if attackers_passed:
                # TODO, add verification of move passed
                chosen_attackers = move
            else:
                xs = list(range(len(eligible_attackers)))
                powerset = list(itertools.chain.from_iterable(itertools.combinations(xs, n) for n in range(len(xs) + 1)))
                element = powerset[move]
                chosen_attackers = [eligible_attackers[i] for i in element]


            self.attackers = chosen_attackers
            for attacker in self.attackers:
                attacker.is_tapped = True
        if self.current_phase_index == Phases.DECLARE_BLOCKERS_STEP:
            blocking_player = self.nonactive_player
            blocking_player.has_blocked = True
            eligible_blockers = blocking_player.get_eligible_blockers(self)
            if len(eligible_blockers) == 0:
                return -1

            if blockers_passed:
                # build dictionary for mapping passed strs to class
                str_to_class = {}
                for blocker in eligible_blockers:
                    str_to_class[blocker.name_id] = blocker
                for attacker in self.attackers:
                    str_to_class[attacker.name_id] = attacker
                # apply blocking assignments
                for attacker_name_id, blocker_name_id_list in move.items():
                    attacker = str_to_class[attacker_name_id]
                    for blocker_name_id in blocker_name_id_list:
                        blocker = str_to_class[blocker_name_id]
                        # append blockers/attackers 
                        attacker.is_blocked_by.append(blocker)
                        blocker.is_blocking.append(attacker)
                        self.blockers.append(blocker)

            else:
                all_blocking_assignments = list(range(np.power(len(self.attackers) + 1, len(eligible_blockers))))
                reshaped_assignments = np.reshape(all_blocking_assignments,
                                                    ([len(self.attackers) + 1] * len(eligible_blockers)))
                blocking_assignments = np.argwhere(reshaped_assignments == move)[0]
                for i in range(len(blocking_assignments)):
                    if blocking_assignments[i] != len(self.attackers):
                        self.attackers[blocking_assignments[i]].is_blocked_by.append(eligible_blockers[i])
                        eligible_blockers[i].is_blocking.append(self.attackers[blocking_assignments[i]])
                        self.blockers.append(eligible_blockers[i])
        # for each attacker that’s become blocked, the active player announces the damage assignment order
        if self.current_phase_index == Phases.DECLARE_BLOCKERS_STEP_509_2:
            for i in range(len(self.attackers)):
                if len(self.attackers[i].is_blocked_by) != 0:
                    if len(self.attackers[i].damage_assignment_order) == 0:
                        self.attackers[i].set_damage_assignment_order(move, direct_order_passed=assignments_passed)
                        return 1
            return -1
        # A blocked creature assigns its combat damage to the creatures blocking it
        if self.current_phase_index == Phases.COMBAT_DAMAGE_STEP_510_1c:
            self.assign_damage_deterministically(player,
                                                 self.attackers[self.attacker_counter], self.blocker_counter, move)
            self.blocker_counter += 1
            if self.blocker_counter >= len(self.attackers[self.attacker_counter].is_blocked_by):
                self.blocker_counter = 0
                self.attacker_counter += 1
            # return all_done

    def assign_damage_deterministically(self, player, attacker, index, amount):
        attacker.assign_damage(index, amount)
        return attacker.damage_to_assign > 0

    # NOTE: this function might be too specialized when more spells than 8ed have been added

    def get_tapped_creature_indices(self):
        tapped_creature_indices = []
        for i in range(len(self.battlefield)):
            if isinstance(self.battlefield[i], Creature):
                if self.battlefield[i].is_tapped:
                    tapped_creature_indices.append(i)
        return tapped_creature_indices

    def get_land_indices(self):
        land_indices = []
        for i in range(len(self.battlefield)):
            if isinstance(self.battlefield[i], Land):
                land_indices.append(i)
        return land_indices

    def get_battlefield_creatures(self):
        creatures = []
        for i in range(len(self.battlefield)):
            if isinstance(self.battlefield[i], Creature):
                creatures.append(self.battlefield[i])
        return creatures

    def get_card_names_from_indices(self, indices):
        card_names = []
        for index in indices:
            permanent = self.battlefield[index]
            if isinstance(permanent, Creature):
                card_names.append(permanent.name_id)
            else:
                card_names.append(str(permanent))
        return card_names

    def player_target_to_string(self):
        actions = [0, 1]
        active_player = self.player_with_priority.index
        actions_str = [None] * 2
        if active_player == 0:
            actions_str[0] = "self"
            actions_str[1] = "opponent"
        else: 
            actions_str[1] = "opponent"
            actions_str[0] = "self"
        return actions_str

    # this function will be called within get_legal_moves
    # to convert a string/language encoded move to a numerical index-based move
    def string_move_to_numerical(traditional_moves, string_moves, string_move):
        '''
            Special cases:
                get_activated_abilities used by: 
                    playable_indices = player.get_playable_cards(self)
                    playable_strings = player.get_playable_card_strings(self)
                    _, ability_indices = player.get_activated_abilities(self)
                    non_passing_moves = list(range(len(playable_indices) + sum(ability_indices)))
                numeric rep = [ 0,1,2,3           4,5,6,7,8]
                               ^- playing a card   ^-activating an ability (maybe multiple per card in the future)
                string rep = ["Forest","Island",....             ability_1_Forest, ability_1_Island (tap_Island) ]
                map: ability_1_Forest -> 4
        '''
        pass

    # NOTE: this function might have become too crowded, consider refactoring
    def get_legal_moves(self, player):
        if self.is_over():
            return [], ["Pass"]
        if player.generic_debt > 0:
            # TODO optimize this to re-use combinations
            mp_as_list = player.get_mp_as_list()
            return list(itertools.combinations(mp_as_list, player.generic_debt)), ManaActionUnroller(self, player)
        if player.casting_spell != "":
            # logging.debug("Returning a spell move now")
            if player.casting_spell == "Vengeance": 
                indices = self.get_tapped_creature_indices()
                return indices, self.get_card_names_from_indices(indices)
            if player.casting_spell == "Stone Rain":
                indices = self.get_land_indices()
                return indices, self.get_card_names_from_indices(indices)
            if player.casting_spell == "Index": # MAYBE REMOVE FOR SIMPLICITY # TODO
                return list(itertools.permutations(list(range(min(5, len(player.deck)))))), None
            if player.casting_spell == "Lava Axe":
                return [0, 1], self.player_target_to_string()
            if player.casting_spell == "Volcanic Hammer":
                self.update_damage_targets()
                return list(range(len(self.damage_targets))), self.damage_target_strings()
            if player.casting_spell == "Sacred Nectar":
                return ["Resolve Spell"], ["Resolve Spell"]
            if player.casting_spell == "Rampant Growth":
                choices = ["Refuse"]
                basic_land_types = ["Plains", "Island", "Swamp", "Mountain", "Forest"]
                for land_type in basic_land_types:
                    if player.find_land_in_library(land_type) >= 0:
                        choices.append(land_type)
                return choices, choices
            return ["Pass"], ["Pass"]
        if self.current_phase_index == Phases.BEGINNING_PHASE:
            return ["Pass"], ["Pass"]
        if self.current_phase_index == Phases.UNTAP_STEP:
            return ["Pass"], ["Pass"]
        if self.current_phase_index == Phases.UPKEEP_STEP:
            return ["Pass"], ["Pass"]
        if self.current_phase_index == Phases.DRAW_STEP:
            return ["Pass"], ["Pass"]
        if self.current_phase_index == Phases.MAIN_PHASE_PRE_COMBAT:
            playable_indices = player.get_playable_cards(self)
            playable_strings = player.get_playable_card_strings(self)
            _, ability_indices = player.get_activated_abilities(self)
            ability_strings = player.get_ability_strings(self)
            non_passing_moves = list(range(len(playable_indices) + sum(ability_indices)))
            non_passing_moves.append("Pass")
            
            # combine strings
            playable_strings.extend(ability_strings)
            playable_strings.append("Pass")
            # return numerical and string representation
            return non_passing_moves,playable_strings  # append the 'pass' move action and return
        if self.current_phase_index == Phases.COMBAT_PHASE:
            return ["Pass"], ["Pass"]
        if self.current_phase_index == Phases.BEGINNING_OF_COMBAT_STEP:
            return ["Pass"], ["Pass"]
        if self.current_phase_index == Phases.DECLARE_ATTACKERS_STEP:
            attacking_player = self.active_player
            if attacking_player.has_attacked or player is not attacking_player:
                return ["Pass"], ["Pass"]
            # next two lines get the power set of attackers
            eligible_attackers = attacking_player.get_eligible_attackers(self)
            xs = list(range(len(eligible_attackers)))
            attacker_combinations = list(itertools.chain.from_iterable(itertools.combinations(xs, n) for n in range(len(xs) + 1)))
            return list(range(
                len(attacker_combinations))), AttackerActionUnroller(self)
        if self.current_phase_index == Phases.DECLARE_BLOCKERS_STEP:
            blocking_player = self.nonactive_player
            if blocking_player.has_blocked or player is not blocking_player:
                return ["Pass"], ["Pass"]
            
            # ActionUnroller if possible blocks
            eligible_blockers = blocking_player.get_eligible_blockers(self)
            if len(eligible_blockers) == 0 or len(self.attackers) == 0:
                alternative_move = ["Pass"]
            else:
                alternative_move = BlockerActionUnroller(self)

            return list(range(np.power(len(self.attackers) + 1, len(eligible_blockers)))), alternative_move
        # for each attacker that’s become blocked, the active player announces the damage assignment order
        if self.current_phase_index == Phases.DECLARE_BLOCKERS_STEP_509_2:
            for i in range(len(self.attackers)):
                if len(self.attackers[i].is_blocked_by) != 0:
                    if len(self.attackers[i].damage_assignment_order) == 0:
                        self.current_attacker = self.attackers[i]
                        return list(range(math.factorial(len(self.attackers[i].is_blocked_by)))), OrderActionUnroller(self)
            return ["Pass"], ["Pass"]

        if self.current_phase_index == Phases.COMBAT_DAMAGE_STEP_510_1c:
            if len(self.attackers) == 0 or self.attacker_counter >= len(self.attackers):
                return ["Pass"], ["Pass"]
            return self.get_possible_damage_assignments(player, self.attackers[self.attacker_counter],
                                                        self.blocker_counter), None
        if self.current_phase_index == Phases.COMBAT_DAMAGE_STEP:
            return ["Pass"], ["Pass"]
        if self.current_phase_index == Phases.END_OF_COMBAT_STEP:
            return ["Pass"], ["Pass"]
        if self.current_phase_index == Phases.MAIN_PHASE_POST_COMBAT:
            return ["Pass"], ["Pass"]
        if self.current_phase_index == Phases.ENDING_PHASE:
            return ["Pass"], ["Pass"]
        if self.current_phase_index == Phases.END_STEP:
            return ["Pass"], ["Pass"]
        if self.current_phase_index == Phases.CLEANUP_STEP:
            return ["Pass"], ["Pass"]

        logging.debug(self.current_phase_index)
        logging.debug("omg we should not have ended up here")

    @staticmethod
    def get_possible_damage_assignments(player, attacker, index):
        if len(attacker.damage_assignment_order) == 0:
            return ["Pass"]
        blocker_i = attacker.damage_assignment_order[index]
        remaining_health = blocker_i.toughness - blocker_i.damage_taken
        if attacker.damage_to_assign < remaining_health or index == len(attacker.damage_assignment_order) - 1:
            return list(range(attacker.damage_to_assign, attacker.damage_to_assign + 1))
        else:
            # modify this line, this is incorrect
            # the attacker does not get to chose how much damage to assign to each after ordering
            #return list(range(remaining_health, attacker.damage_to_assign + 1))
            return [remaining_health]

    def start_game(self):
        self.active_player.passed_priority = False
        self.active_player.can_play_land = True
        for i in range(len(self.players)):
            self.players[i].shuffle_deck()
            for j in range(self.starting_hand_size):
                self.players[i].draw_card()

    def start_new_turn(self):
        self.current_phase_index = Phases.BEGINNING_PHASE
        self.active_player = self.players[1 - self.active_player.index]
        self.player_with_priority = self.active_player
        self.nonactive_player = self.players[1 - self.active_player.index]
        self.active_player.draw_card()
        self.active_player.can_play_land = True
        for permanent in self.battlefield:
            permanent.is_tapped = False
            if isinstance(permanent, Creature):
                permanent.summoning_sick = False
                permanent.damage = 0
        for i in range(len(self.players)):
            self.players[i].reset_mp()
            self.players[i].has_attacked = False
            self.players[i].has_blocked = False

    def go_to_next_phase(self):
        # logging.debug(self.current_phase_index)
        self.current_phase_index = self.current_phase_index.next()

        if self.current_phase_index == Phases.CLEANUP_STEP:
            self.start_new_turn()
            return True
        elif self.current_phase_index == Phases.COMBAT_DAMAGE_STEP:
            if self.apply_combat_damage():
                self.check_state_based_actions()
            self.clean_up_after_combat()
        elif self.current_phase_index == Phases.DECLARE_BLOCKERS_STEP:
            self.nonactive_player.has_passed = False
            self.active_player.has_passed = True
            self.player_with_priority = self.nonactive_player
        else:
            self.nonactive_player.has_passed = True
            self.active_player.has_passed = False
            self.player_with_priority = self.active_player

    def is_over(self):
        for i in range(len(self.players)):
            if self.players[i].has_lost:
                return True
        return False

    def apply_combat_damage(self):
        any_attackers = False
        for permanent in self.battlefield:
            if isinstance(permanent, Creature):
                if permanent in self.attackers:
                    if len(permanent.is_blocked_by) > 0:
                        for i in range(len(permanent.is_blocked_by)):
                            permanent.is_blocked_by[i].take_damage(permanent.damage_assignment[i])
                            permanent.take_damage(permanent.is_blocked_by[i].power)
                    else:
                        permanent.deal_combat_damage_to_opponent(self)
                any_attackers = True
        return any_attackers

    def check_state_based_actions(self):
        # 704.5g
        for permanent in self.battlefield:
            if isinstance(permanent, Creature):
                if permanent.is_dead:
                    self.battlefield.remove(permanent)
                    permanent.owner.graveyard.append(permanent)

    def clean_up_after_combat(self):
        # TODO: Simplify this and test!
        self.attackers = []
        self.blockers = []
        self.attacker_counter = 0
        self.blocker_counter = 0
        for permanent in self.battlefield:
            if isinstance(permanent, Creature):
                # TODO: attribute "is_attacking" seems to be useless, remove this from everywhere
                permanent.is_attacking = []
                permanent.is_blocking = []
                permanent.is_blocked_by = []
                permanent.damage_assignment_order = []
                permanent.damage_assignment = []

# One-use "action unroller" to take a action such as declaring attackers and change it into a multi-step process
# that is compatible with auto-regressive models

class ActionUnroller(ABC):
    def __init__(self, game):
        super().__init__()
        self.game = game
        self.done = False

    @abstractmethod
    def get_legal_moves():
        pass

    @abstractmethod
    def done():
        pass

    @abstractmethod
    def register_move(self, move):
        pass

# Maybe redo to manipulate classes instead of strings
class AttackerActionUnroller(ActionUnroller):
    def __init__(self,game):
        super().__init__(game)
        # This is also computed previously, maybe re-factor to reduce redundancy
        self.eligible_attackers = game.active_player.get_eligible_attackers(game)
        self.attacker_names = [attacker.name_id for attacker in self.eligible_attackers]
        self.legal_moves = self.attacker_names.copy()
        self.legal_moves.append("Pass")
        self.selected_attackers = []

    def done(self):
        return self.done

    def get_legal_moves(self):
        assert (not self.done), "Called get_move when done"
        return self.legal_moves

    # returns state after registering the move 
    def register_move(self, move):
        assert (move in self.legal_moves), "Invalid move"
        if move == "Pass":
            self.done = True
        else:
            self.selected_attackers.append(move)
        self.legal_moves.remove(move)
        return self.game.get_board_string(additional_attacker_names=self.selected_attackers)

    # officially applies the "registered" moves to the game
    # should be called after "done" unrolling
    def make_move(self):
        assert (self.done), "Unrolling not complete"
        attacker_cards = []
        # search through eligible attackers ,
        # if an eligible attacker is in the selected attackers list, append card class to attacker cards

        for attacker in self.eligible_attackers:
            if attacker.name_id in self.selected_attackers:
                attacker_cards.append(attacker)
                self.selected_attackers.remove(attacker.name_id)
        self.game.make_move(move=attacker_cards, attackers_passed=True)

# Maybe redo to manipulate classes instead of strings
class BlockerActionUnroller(ActionUnroller):
    def __init__(self,game):
        super().__init__(game)
        self.blocking_player = self.game.nonactive_player
        self.eligible_blockers = self.blocking_player.get_eligible_blockers(self.game)
        self.num_blockers = len(self.eligible_blockers)
        self.num_attackers = len(self.game.attackers)
        self.blocker_index = 0
        assert(self.game.num_blockers > 0 and self.game.num_attackers > 0), "BlockerActionUnroller called with zero attackers or blockers"
        
        self.block_assignment_dict = {} # key is attacker, value is an array of blockers
        self.current_legal_moves = None


    def done(self):
        return self.done

    def get_legal_moves(self):
        assert (not self.done), "Called get_move when done"
        # might want to verify this line ->, refactor
        moves = ["Pass"]
        if len(self.blocker_index + 1 > self.num_blockers):
            pass
        else:
            current_blocker = self.eligible_blockers[self.blocker_index]
            for attacker in self.game.attackers:
                moves.append(f'{current_blocker.name_id}$block${attacker.name_id}')
        self.current_legal_moves = moves
        return moves

    # returns state after registering the move 
    def register_move(self, move):
        assert (move in self.current_legal_moves), "Invalid move"
        
        # parse move
        info = move.split('block')
        # get attacker and blocker names, remove middle $s
        blocker_name_id = info[0][:-1]
        attacker_name_id = info[1][1:]

        # add to dictionary
        if attacker_name_id in self.block_assignment_dict:
            self.block_assignment_dict[attacker_name_id].append(blocker_name_id)
        else:
            self.block_assignment_dict[attacker_name_id] = [blocker_name_id]

        
        self.blocker_index += 1
        # check for done
        if self.blocker_index >= self.num_blockers:
            self.done = True
        return self.game.get_board_string(additional_block_assignments=self.block_assignment_dict) # make sure to pass updated blocker info

    # officially applies the "registered" moves to the game
    # should be called after "done" unrolling
    def make_move(self):
        assert (self.done), "Unrolling not complete"
        
        self.game.make_move(move=self.block_assignment_dict, blockers_passed=True)
        return self.game.get_board_string(additional_block_assignments=self.block_assignment_dict))

class OrderActionUnroller(ActionUnroller):
    def __init__(self,game):
        super().__init__(game)
        self.attacking_player = self.game.active_player
        self.attacker = self.game.current_attacker
        self.blockers = self.attacker.is_blocked_by
        self.blocker_names = [blocker.name_id for blocker in self.blockers]
        self.legal_blocker_names = self.blocker_names.copy() # blockers yet selected
        self.selected_blocker_names = []
        assert(self.game.num_blockers > 0 and self.game.num_attackers > 0 and len(self.blockers) > 0), "OderActionUnroller called with zero attackers or blockers"
        
        self.current_legal_moves = None


    def done(self):
        return self.done

    def get_legal_moves(self):
        assert (not self.done), "Called get_move when done"
        assert (len(self.selected_blocker_names < len(blocker_names)))

        return self.legal_blocker_names

    def get_info(self):
        info = {
            "attacker_name": [self.attacker.name_id]
            "blocker_names": self.selected_blocker_names
        }
        return info

    # returns state after registering the move 
    def register_move(self, move):
        assert (move in self.legal_blocker_names), "Invalid move"
        
        # add to selected moves  and remove from legal
        self.selected_blocker_names.append(move)
        self.legal_blocker_names.remove(move)
        
        # if just one decision left, we can make it, otherwise just mark as done
        if len(self.legal_blocker_names) <= 1:
            self.done = True
            if len(self.legal_blocker_names) == 1:
                self.selected_blocker_names.append(self.legal_blocker_names.pop())

        
        return self.game.get_board_string(additional_order_assignments=self.get_info())

    # officially applies the "registered" moves to the game
    # should be called after "done" unrolling
    def make_move(self):
        assert (self.done), "Unrolling not complete"
        ordered_blockers = []
        # convert selected names to indexes
        for blocker_name in self.selected_blocker_names:
            ordered_blockers.append(self.blockers[self.blocker_names.index(blocker_name)])
        # passes the order of blockers 
        self.game.make_move(move=ordered_blockers, assignments_passed=True)
        return self.game.get_board_string(additional_order_assignments=self.get_info()))

class ManaActionUnroller(ActionUnroller):
    def __init__(self,game, player):
        super().__init__(game)
        self.player = player
        self.mp_as_list = player.get_mp_as_list()
        self.unused_mana = mp_as_list.copy()
        self.used_mana = []
        assert (player.generic_debt > 0), "Created ManaActionUnroller with no debt to pay"
        self.combinations = list(itertools.combinations(self.mp_as_list, player.generic_debt))      
        self.current_legal_moves = None

    def done(self):
        return self.done

    def get_info(self):
        debt = player.generic_debt - len(self.used_mana)
        assert (debt >= 0), "invalid debt info returned"
        manapool = self.player.manapool.copy()
        for mana in self.used_mana:
            assert(manapool[mana] > 0), "invalid manapool info returned"
            manapool[mana] -= 1
        info = {
            'debt': debt
            'manapool': manapool
        }
        return info

    def get_legal_moves(self):
        assert (not self.done), "Called get_move when done"
        return list(set(self.unused_mana))

    # returns state after registering the move 
    def register_move(self, move):
        assert (move in self.get_legal_moves()), "Invalid move"
        
        self.used_mana.append(move)
        self.unused_mana.remove(move)
        
        if len(self.used_mana) >= player.generic_debt:
            self.done = True
        
        return self.game.get_board_string(additional_mana_info=self.get_info()) 

    # officially applies the "registered" moves to the game
    # should be called after "done" unrolling
    def make_move(self):
        assert (self.done), "Unrolling not complete"
        
        self.game.make_move(move=self.used_mana)
        return self.game.get_board_string(additional_mana_info=self.get_info())



