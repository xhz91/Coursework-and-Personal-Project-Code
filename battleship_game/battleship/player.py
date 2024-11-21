import random

from battleship.board import Board
from battleship.convert import CellConverter

class Player:
    """ Class representing the player
    """
    count = 0  # for keeping track of number of players
    
    def __init__(self, board=None, name=None):
        """ Initialises a new player with its board.

        Args:
            board (Board): The player's board. If not provided, then a board
                will be generated automatically
            name (str): Player's name
        """
        
        if board is None:
            self.board = Board()
        else:
            self.board = board
        
        Player.count += 1
        if name is None:
            self.name = f"Player {self.count}"
        else:
            self.name = name
    
    def __str__(self):
        return self.name
    
    def select_target(self):
        """ Select target coordinates to attack.
        
        Abstract method that should be implemented by any subclasses of Player.
        
        Returns:
            tuple[int, int] : (x, y) cell coordinates at which to launch the 
                next attack
        """
        raise NotImplementedError
    
    def receive_result(self, is_ship_hit, has_ship_sunk):
        """ Receive results of latest attack.
        
        Player receives notification on the outcome of the latest attack by the 
        player, on whether the opponent's ship is hit, and whether it has been 
        sunk. 
        
        This method does not do anything by default, but can be overridden by a 
        subclass to do something useful, for example to record a successful or 
        failed attack.
        
        Returns:
            None
        """
        return None
    
    def has_lost(self):
        """ Check whether player has lost the game.
        
        Returns:
            bool: True if and only if all the ships of the player have sunk.
        """
        return self.board.have_all_ships_sunk()


class ManualPlayer(Player):
    """ A player playing manually via the terminal
    """
    def __init__(self, board, name=None):
        """ Initialise the player with a board and other attributes.
        
        Args:
            board (Board): The player's board. If not provided, then a board
                will be generated automatically
            name (str): Player's name
        """
        super().__init__(board=board, name=name)
        self.converter = CellConverter((board.width, board.height))
        
    def select_target(self):
        """ Read coordinates from user prompt.
               
        Returns:
            tuple[int, int] : (x, y) cell coordinates at which to launch the 
                next attack
        """
        print(f"It is now {self}'s turn.")

        while True:
            try:
                coord_str = input('coordinates target = ')
                x, y = self.converter.from_str(coord_str)
                return x, y
            except ValueError as error:
                print(error)


class RandomPlayer(Player):
    """ A Player that plays at random positions.

    However, it does not play at the positions:
    - that it has previously attacked
    """
    def __init__(self, name=None):
        """ Initialise the player with an automatic board and other attributes.
        
        Args:
            name (str): Player's name
        """
        # Initialise with a board with ships automatically arranged.
        super().__init__(board=Board(), name=name)
        self.tracker = set()

    def select_target(self):
        """ Generate a random cell that has previously not been attacked.
        
        Also adds cell to the player's tracker.
        
        Returns:
            tuple[int, int] : (x, y) cell coordinates at which to launch the 
                next attack
        """
        target_cell = self.generate_random_target()
        self.tracker.add(target_cell)
        return target_cell

    def generate_random_target(self):
        """ Generate a random cell that has previously not been attacked.
               
        Returns:
            tuple[int, int] : (x, y) cell coordinates at which to launch the 
                next attack
        """
        has_been_attacked = True
        random_cell = None
        
        while has_been_attacked:
            random_cell = self.get_random_coordinates()
            has_been_attacked = random_cell in self.tracker

        return random_cell

    def get_random_coordinates(self):
        """ Generate random coordinates.
               
        Returns:
            tuple[int, int] : (x, y) cell coordinates at which to launch the 
                next attack
        """
        x = random.randint(1, self.board.width)
        y = random.randint(1, self.board.height)
        return (x, y)


class AutomaticPlayer(Player):
    """ Player playing automatically using a strategy.
        
        Detailed strategy:
        1. At the beginning of the game, we keep attacking random cells until there's a hit (i.e. until we get the first is_ship_hit == True)
        2. Once there is a successful hit, attack a cell adjacent to the successful hit cell (can be up/down/left/right, random choice among these 4 directions). 
            - Continues trying all the adjacent cells until a hit
            - Then follow the direction of the two successful hits (horizontal/vertical), attack a cells to the left or the right on the same ship
                e.g. if C8 and C9 both returns is_ship_hit == True, then my next target_cell will be either C7 or C10. IF C7 result in is_ship_hit == False, then try C10. Then try C11, etc.
            - Continue this until the ship is sunk
        3. If a ship has sunk, then I will avoid all the cells near the ship as there won't be any other ships near this sunk ship.
        4. Once this ship sinks, we will start attcking random cells again until there's a hit. Essentially repeating step 1-3 above
    """
    def __init__(self, name=None):
        """ Initialise the player with an automatic board and other attributes.
        
        Args:
            name (str): Player's name
        """
        # Initialise with a board with ships automatically arranged.
        super().__init__(board=Board(), name=name)
        
        # TODO: Add any other attributes necessary for your strategic player
        self.tracker = {}                  # {target_cell: is_ship_hit} tracks cells that have been attacked already and the results of the attack
        self.successful_hit_cells = []     # tracks the cells that results in a successful hit, i.e. marked with X
        self.sink_ship_cells = set()       # set of cells that is near sink ships
        
        self.current_target_cell = None    # tracks the current target cell, the result of this attack will be recoded via receive_result method
        self.target_direction = None       # direction to continue attacking


    def select_target(self):
        """ Using the result of attacks collected via receive_result method, make decision on the next target coordinates to attack.
        
        Returns:
            tuple[int, int] : (x, y) cell coordinates at which to launch the 
                next attack
        """
        # TODO: Complete this method

        # When we don't have a successful hit (i.e. a 'X') on the board, attack random cells. This could be at the beginning of the game when no attacks has been done yet or when a ship has just sunk
        if len(self.successful_hit_cells) == 0:
            target_cell = self.generate_random_target()

        # Once we have a successful hit, then we pick next target to be one of its adjacent cells
        elif len(self.successful_hit_cells) == 1:
            target_cell = self.generate_adjacent_target(self.successful_hit_cells[0])
            
            # If an adjacent target can't be found, e.g. because the cell is already at the boundary of board, choose another cell randomly as target
            if target_cell is None:
                self.successful_hit_cells.clear()
                target_cell = self.generate_random_target()
        
        # If we have more hits, then we follow the same direction to identify cells within the same ship
        else:
            target_cell = self.continue_in_direction()
            
            # If cells on the same ship can't be find, choose target randomly
            if target_cell is None:
                self.successful_hit_cells.clear()
                target_cell = self.generate_random_target()
        
        self.current_target_cell = target_cell
        return target_cell


    def receive_result(self, is_ship_hit, has_ship_sunk):
        """ Collect the result of the latest attack. This method forms the base of the decision about next target cell, which is made in select_target method.

        Args:
            is_ship_hit (bool): True if the attack hits a ship
            has_ship_sunk: True if the attack resulted in sinking a ship

        Returns:
            None
        """
        # Record the current target cell and the result of the attack (given by input arguments) into self.tracker
        if self.current_target_cell is not None:
            self.tracker[self.current_target_cell] = is_ship_hit

        # If target cell gives a successful hit, then first add the cell to self.successful_hit_cells, so that we can select it's adjacent cells as next targets.
        if is_ship_hit:
            self.successful_hit_cells.append(self.current_target_cell)

            # If this success results in two consecutive cells with successful hit, then we set the direction of next target to be the same as these two cells.
            if len(self.successful_hit_cells) >= 2:
                self.target_direction = self.determine_direction()
        
        # If the ship has sunk, then we need to find all the cells near the sunk ship and rule them out from target pool. Because no ships will be near another ship on the board as per rule of the game.
        if has_ship_sunk:
            for cell in self.successful_hit_cells:
                self.sink_ship_cells.update(self.get_nearby_cells(cell))
            
            # Reset successful_hit_cells list and target_direction so we can start next round of random exploration until the next hit
            self.successful_hit_cells.clear()
            self.target_direction = None
                
        return None


    def generate_random_target(self):
        """ Generate a random cell that has previously not been attacked and not near a sunk ship
               
        Returns:
            tuple[int, int] : (x, y) cell coordinates at which to launch the 
                next attack
        """
        has_been_attacked = True
        is_near_sunk_ship = True
        random_cell = None
        
        while has_been_attacked or is_near_sunk_ship:
            random_cell = self.get_random_coordinates()
            has_been_attacked = random_cell in self.tracker
            is_near_sunk_ship = random_cell in self.sink_ship_cells

        return random_cell


    def get_random_coordinates(self):
        """ Generate random coordinates.
               
        Returns:
            tuple[int, int] : (x, y) cell coordinates at which to launch the 
                next attack
        """
        x = random.randint(1, self.board.width)
        y = random.randint(1, self.board.height)
        return (x, y)


    def generate_adjacent_target(self, cell):
        """ Generate a cell that is adjacent(up/down/left/right) a cell and has previously not been attacked and not near a sunk ship
               
        Returns:
            tuple[int, int] : (x, y) cell coordinates at which to launch the 
                next attack
        """
        x, y = cell
        possible_coordinates = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        valid_coordinates = []
        
        for coordinate in possible_coordinates:
            if (self.is_within_board_bound(coordinate) and
                coordinate not in self.tracker and
                coordinate not in self.sink_ship_cells):
                valid_coordinates.append(coordinate)
        
        if valid_coordinates:
            return random.choice(valid_coordinates)
        else:
            return None

    
    def determine_direction(self):
        """ If there are two or more consecutive cells with successful hit, then identify whether they are on a horizontal ship or a vertical ship
            The method updates the self.target_direction attribute value to be 'vertical' or 'horizontal'
        """
        (x1, y1) = self.successful_hit_cells[0]
        (x2, y2) = self.successful_hit_cells[1]
        
        if x1 == x2:
            return 'vertical'

        elif y1 == y2:
            return 'horizontal'


    def continue_in_direction(self):
        """ Generate a cell that is on the same ship with two or cells with successful hit. e.g. the function will return C7 or C10 if C8 & C9 are already marked with 'X'
               
        Returns:
            tuple[int, int] : (x, y) cell coordinates at which to launch the 
                next attack
        """
        first_hit = self.successful_hit_cells[0]
        last_hit = self.successful_hit_cells[-1]
        
        if self.target_direction == 'horizontal':
            x_left = min(first_hit[0], last_hit[0]) - 1
            x_right = max(first_hit[0], last_hit[0]) + 1
            y = first_hit[1]
            left_candidate = (x_left,y)
            right_candidate = (x_right,y)

            if (self.is_within_board_bound(left_candidate) and
                    left_candidate not in self.tracker and
                    left_candidate not in self.sink_ship_cells):
                return left_candidate
            elif (self.is_within_board_bound(right_candidate) and
                    right_candidate not in self.tracker and
                    right_candidate not in self.sink_ship_cells):
                return right_candidate
        
        elif self.target_direction == 'vertical':
            x = first_hit[0]
            y_up = min(first_hit[1], last_hit[1]) - 1
            y_down = max(first_hit[1], last_hit[1]) + 1
            up_candidate = (x,y_up)
            down_candidate = (x,y_down)

            if (self.is_within_board_bound(up_candidate) and
                    up_candidate not in self.tracker and
                    up_candidate not in self.sink_ship_cells):
                return up_candidate
            elif (self.is_within_board_bound(down_candidate) and
                    down_candidate not in self.tracker and
                    down_candidate not in self.sink_ship_cells):
                return down_candidate

        return None


    def is_within_board_bound(self, cell):
        """Check if a cell is within the board bound

        Args:
        cell (tuple): (x,y) coordinates of a cell

        Returns:
        True if the cell is within the board width and height, False if otherwise
        """
        x, y = cell
        return (1 <= x <= self.board.width) and (1 <= y <= self.board.height)


    def get_nearby_cells(self, cell):
        """ Find all the cells surrounding a cell
        
        Returns: 
            list of all cells surrounding a cell
        """
        x, y  = cell
        nearby_cells = [(x-1, y),(x+1, y),(x, y-1),(x, y+1),(x-1, y-1),(x-1, y+1),(x+1, y-1),(x+1, y+1)]
        valid_cells = []
        for cell in nearby_cells:
            if self.is_within_board_bound(cell):
                valid_cells.append(cell)
        return valid_cells