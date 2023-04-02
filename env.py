from deck import *
from game import *
from player import *
from phases import Phases
import copy



class MtgEnv:
    # initialize environment
    def __init__(self, deck_1, deck_name_1, deck_2, deck_name_2):
        self.deck_1 = deck_1
        self.deck_name_1 = deck_name_1
        self.deck_2 = deck_2
        self.deck_name_2 = deck_name_2

        self.player1 = None
        self.player2 = None

        # move strings and indice
        # agent returns a string, which is converted to an index before game.move()
        self.move_indices = None
        self.move_strings = None
        self.state = None
        self.active_player = None

        # for combat-multi-step actions
        self.action_unroller = None

        # initlize game
        self.reset()
    
    def state_action_to_query(self, state,available_actions):
        available_actions = ', '.join(available_actions)
        query = f'{state}\navailable_actions: {available_actions}\n'
        return query
    def format_action(self, action):
        return f'best: {action}'

    # return index of player with priority
    @property
    def other_player_index(self):
        if self.active_player_index is None:
            return None
        return 1 - self.active_player_index
    
    def reset(self):
        # create players
        self.player1 = Player(self.deck_1(), name='Player0', deck_name=self.deck_name_1)
        self.player2 = Player(self.deck_2(), name='Player1', deck_name=self.deck_name_2)

        self.game = Game([self.player1, self.player2])
        self.game.start_game()

        self.active_player_index = self.game.player_with_priority.index
        self.move_indices, self.move_strings = self.game.get_moves()
        self.state = self.game.get_board_string()

        # return state and valid actions
        possible_moves = copy.deepcopy(self.move_strings)
        state = copy.deepcopy(self.state)
        return possible_moves, state, self.active_player_index

    def step(self, action_str, passed_index=False):
        # check that action is valid
        assert action_str in self.move_strings, f"Invalid action {action_str} not in {self.move_strings}"
        # TODO, handle dmage step

        # check if currently in an action unroller loop
        if self.action_unroller is not None:
            assert not passed_index, "should not pass index to unroller"
            self.state = self.action_unroller.register_move(action_str)
            # Unroller finished, apply action and get new state
            if self.action_unroller.is_done():
                self.action_unroller.make_move()
                self.action_unroller = None

                # update state
                self.state = self.game.get_board_string()
                move_indices, move_strings = self.game.get_moves()
                if isinstance(move_strings, ActionUnroller):
                    self.action_unroller = move_strings
                    self.move_strings = self.action_unroller.get_legal_moves()
                    self.move_indices = None
                else:
                    self.move_strings = move_strings
                    self.move_indices = move_indices

            else:
                # Still in unroller, move indices not needed as unroller takes action strings
                self.move_strings = self.action_unroller.get_legal_moves()
                self.move_indices = None
        else:
            # make move
            if passed_index:
                move_index = action_str
            else:
                move_index = self.move_indices[self.move_strings.index(action_str)]
            
            self.game.make_move(move_index, False)
            # get next moves
            move_indices, move_strings = self.game.get_moves()
            self.state = self.game.get_board_string()
            if isinstance(move_strings, ActionUnroller):
                self.action_unroller = move_strings
                self.move_strings = self.action_unroller.get_legal_moves()
                self.move_indices = None
            else:
                self.move_strings = move_strings
                self.move_indices = move_indices

        if self.game.current_phase_index == Phases.COMBAT_DAMAGE_STEP_510_1c:
            state, possible_moves, active_player_index, done, info = self.step(self.move_indices[0], passed_index=True)
        else:
            state = copy.deepcopy(self.state)
            possible_moves = copy.deepcopy(self.move_strings)
            self.active_player_index = self.game.player_with_priority.index
            active_player_index = self.active_player_index
            info = {}
            done = self.game.is_over()
            if self.game.players[0].has_lost:
                info['winning_player'] = 1
            else:
                info['winning_player'] = 0

        return state, possible_moves, active_player_index, done, info
    
if __name__ == '__main__':
    # create environment
    env = MtgEnv(get_8ed_core_gold_deck, 'Gold0', get_8ed_core_gold_deck, 'Gold1')
    done = False
    possible_moves, state, active_player_index = env.reset()
    while not done:
        print(f"Player {active_player_index} turn")
        print(state)
        print(possible_moves)
        # input action index
        if len(possible_moves) == 1:
            action = 0
        else:
            action = int(input("Enter action: "))
        action = possible_moves[action]
        state, possible_moves, active_player_index, done, info = env.step(action)
