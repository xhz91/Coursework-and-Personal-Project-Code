import random

from battleship.convert import CellConverter

class Ship:
    """ Represent a ship that is placed on the board.
    """
    def __init__(self, start, end, should_validate=True):
        """ Creates a ship given its start and end coordinates on the board. 
        
        The order of the cells do not matter.

        Args:
            start (tuple[int, int]): tuple of 2 positive integers representing
                the starting cell coordinates of the Ship on the board
            end (tuple[int, int]): tuple of 2 positive integers representing
                the ending cell coordinates of the Ship on the board
            should_validate (bool): should the constructor check whether the 
                given coordinates result in a horizontal or vertical ship? 
                Defaults to True.

        Raises:
            ValueError: if should_validate==True and 
                if the ship is neither horizontal nor vertical
        """
        # Start and end (x, y) cell coordinates of the ship
        self.x_start, self.y_start = start
        self.x_end, self.y_end = end

        # make x_start on left and x_end on right
        self.x_start, self.x_end = (
            min(self.x_start, self.x_end), max(self.x_start, self.x_end)
        )
        
        # make y_start on top and y_end on bottom
        self.y_start, self.y_end = (
            min(self.y_start, self.y_end), max(self.y_start, self.y_end)
        )
        
        if should_validate:
            if not self.is_horizontal() and not self.is_vertical():
                raise ValueError("The given coordinates are invalid. "
                    "The ship needs to be either horizontal or vertical.")

        # Set of all (x,y) cell coordinates that the ship occupies
        self.cells = self.get_cells()
        
        # Set of (x,y) cell coordinates of the ship that have been damaged
        self.damaged_cells = set()
    
    def __len__(self):
        return self.length()
        
    def __repr__(self):
        return (f"Ship(start=({self.x_start},{self.y_start}), "
            f"end=({self.x_end},{self.y_end}))")
        
    def is_vertical(self):
        """ Check whether the ship is vertical.
        
        Returns:
            bool : True if the ship is vertical. False otherwise.
        """
        # TODO: Complete this method
        if self.x_start == self.x_end:
            return True
        return False
   
    def is_horizontal(self):
        """ Check whether the ship is horizontal.
        
        Returns:
            bool : True if the ship is horizontal. False otherwise.
        """
        # TODO: Complete this method
        if self.y_start == self.y_end:
            return True
        return False
    
    def get_cells(self):
        """ Get the set of all cell coordinates that the ship occupies.
        
        For example, if the start cell is (3, 3) and end cell is (5, 3),
        then the method should return {(3, 3), (4, 3), (5, 3)}.
        
        This method is used in __init__() to initialise self.cells
        
        Returns:
            set[tuple] : Set of (x ,y) coordinates of all cells a ship occupies
        """
        # TODO: Complete this method
        cell_list = [(x,y) for x in range(self.x_start, self.x_end+1) for y in range(self.y_start, self.y_end+1)]
        return set(cell_list)

    def length(self):
        """ Get length of ship (the number of cells the ship occupies).
        
        Returns:
            int : The number of cells the ship occupies
        """
        # TODO: Complete this method
        return len(self.get_cells())

    def is_occupying_cell(self, cell):
        """ Check whether the ship is occupying a given cell

        Args:
            cell (tuple[int, int]): tuple of 2 positive integers representing
                the (x, y) cell coordinates to check

        Returns:
            bool : return True if the given cell is one of the cells occupied 
                by the ship. Otherwise, return False
        """
        # TODO: Complete this method
        if cell in self.get_cells():
            return True
        return False
    
    def receive_damage(self, cell):
        """ Receive attack at given cell. 
        
        If ship occupies the cell, add the cell coordinates to the set of 
        damaged cells. Then return True. 
        
        Otherwise return False.

        Args:
            cell (tuple[int, int]): tuple of 2 positive integers representing
                the cell coordinates that is damaged

        Returns:
            bool : return True if the ship is occupying cell (ship is hit). 
                Return False otherwise.
        """
        # TODO: Complete this method
        if self.is_occupying_cell(cell):
            self.damaged_cells.add(cell)
            return True
        return False
    
    def count_damaged_cells(self):
        """ Count the number of cells that have been damaged.
        
        Returns:
            int : the number of cells that are damaged.
        """
        # TODO: Complete this method
        return len(self.damaged_cells)
        
    def has_sunk(self):
        """ Check whether the ship has sunk.
        
        Returns:
            bool : return True if the ship is damaged at all its positions. 
                Otherwise, return False
        """
        # TODO: Complete this method
        all_cell_damaged = all(cell in self.damaged_cells for cell in self.cells)
        return all_cell_damaged
    
    def is_near_ship(self, other_ship):
        """ Check whether a ship is near another ship instance.
        
        Hint: Use the method is_near_cell(...) to complete this method.

        Args:
            other_ship (Ship): another Ship instance against which to compare

        Returns:
            bool : returns True if and only if the coordinate of other_ship is 
                near to this ship. Returns False otherwise.
        """
        # TODO: Complete this method
        other_ship_near_any_self_cells =  any(other_ship.is_near_cell(cell) for cell in self.cells)
        return other_ship_near_any_self_cells


    def is_near_cell(self, cell):
        """ Check whether the ship is near an (x,y) cell coordinate.

        In the example below:
        - There is a ship of length 3 represented by the letter S.
        - The positions 1, 2, 3 and 4 are near the ship
        - The positions 5 and 6 are NOT near the ship

        --------------------------
        |   |   |   |   | 3 |   |
        -------------------------
        |   | S | S | S | 4 | 5 |
        -------------------------
        | 1 |   | 2 |   |   |   |
        -------------------------
        |   |   | 6 |   |   |   |
        -------------------------

        Args:
            cell (tuple[int, int]): tuple of 2 positive integers representing
                the (x, y) cell coordinates to compare

        Returns:
            bool : returns True if and only if the (x, y) coordinate is at most
                one cell from any part of the ship OR is at the corner of the 
                ship. Returns False otherwise.
        """
        return (self.x_start-1 <= cell[0] <= self.x_end+1 
                and self.y_start-1 <= cell[1] <= self.y_end+1)


class ShipFactory:
    """ Class to create new ships in specific configurations."""
    def __init__(self, board_size=(10,10), ships_per_length=None):
        """ Initialises the ShipFactory class with necessary information.
        
        Args: 
            board_size (tuple[int,int]): the (width, height) of the board in 
                terms of number of cells. Defaults to (10, 10)
            ships_per_length (dict): A dict with the length of ship as keys and
                the count as values. Defaults to 1 ship each for lengths 1-5.
        """
        self.board_size = board_size
        
        if ships_per_length is None:
            # Default: lengths 1 to 5, one ship each
            self.ships_per_length = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1}
        else:
            self.ships_per_length = ships_per_length

    @classmethod
    def create_ship_from_str(cls, start, end, board_size=(10,10)):
        """ A class method for creating a ship from string based coordinates.
        
        Example usage: ship = ShipFactory.create_ship_from_str("A3", "C3")
        
        Args:
            start (str): starting coordinate of the ship (example: 'A3')
            end (str): ending coordinate of the ship (example: 'C3')
            board_size (tuple[int,int]): the (width, height) of the board in 
                terms of number of cells. Defaults to (10, 10)

        Returns:
            Ship : a Ship instance created from start to end string coordinates
        """
        converter = CellConverter(board_size)
        return Ship(start=converter.from_str(start),
                    end=converter.from_str(end))

    def generate_ships(self):
        """ Generate a list of ships in the appropriate configuration.
        
        The number and length of ships generated must obey the specifications 
        given in self.ships_per_length.
        
        The ships must also not overlap with each other, and must also not be 
        too close to one another (as defined earlier in Ship::is_near_ship())
        
        The coordinates should also be valid given self.board_size
        
        Returns:
            list[Ships] : A list of Ship instances, adhering to the rules above
        """
        # TODO: Complete this method
        ships = []
        total_ships = sum(self.ships_per_length.values())  # Total number of ships
        
        # Get a shuffled list of ship lengths. This is all the ships that needs to be placed, shuffled to add randomness
        # e.g. [1, 5, 3, 2, 4] or [1, 3, 4, 5, 2, 3]
        all_length = []
        for length in self.ships_per_length:
            for count in range(self.ships_per_length[length]):
                all_length.append(length)
        random.shuffle(all_length)

        # Main loop for placing ship on board
        for length_item in all_length:

            # For each ship item that needs to be placed, try randomly placing this ship for 100 times, return the first successful placement.            
            max_attempt = 100
            for attempt in range(max_attempt):
                # Randomly generate staring point and direction
                x_start = random.randint(1, self.board_size[0])
                y_start = random.randint(1, self.board_size[1])
                starting_point = (x_start, y_start)
                direction = random.choice(['horizontal','vertical'])

                # Calculate the end point from start point and direction
                if direction == 'horizontal':
                    end_point = (x_start + length_item - 1, y_start)
                elif direction == 'vertical':
                    end_point = (x_start, y_start + length_item - 1)

                if self.is_valid_placement(starting_point, end_point, ships):
                    ship = Ship(starting_point, end_point)
                    ships.append(ship)
                    placed = True
                    break

        return ships


    def is_valid_placement(self, starting_point, end_point, existing_ship):
        """
        Check if the placement of a ship is valid - whether it is within the board bounds and not near or overlap other existing ships
        
        Args:
        starting_point (tuple): the (x,y) coordinate of the start of the ship
        end_point (tuple): the (x,y) coordinate of the end of the ship
        direction (str): the direction the ship is placed. Either horizontal or vertical
        existing_ship (list): the list of Ship instances that are already placed on the board
        
        Returns:
        is_valid (boolean): return True if the new ship is withinthe board bounds and is not near or overlap other exiting ships, False otherwise
        """
        x_start, y_start = starting_point
        x_end, y_end = end_point
        
        # Returns True if all coordinates are within bound, False if one or more coordinates are outside bound
        if (1 <= x_start <= self.board_size[0] and
            1 <= x_end <= self.board_size[0] and
            1 <= y_start <= self.board_size[1] and
            1 <= y_end <= self.board_size[1]
           ):
            within_bound = True
        else:
            within_bound = False
        
        # Returns True if ship_of_test is near any existing ships
        ship_of_test = Ship(starting_point, end_point)
        ship_near_other = any(ship_of_test.is_near_ship(other_ship) for other_ship in existing_ship)

        return within_bound and not ship_near_other

            
        
        
if __name__ == '__main__':
    # SANDBOX for you to play and test your methods

    ship = Ship(start=(3, 3), end=(5, 3))
    print(ship.get_cells())
    print(ship.length())
    print(ship.is_horizontal())
    print(ship.is_vertical())
    print(ship.is_near_cell((5, 3)))
    
    print(ship.receive_damage((4, 3)))
    print(ship.receive_damage((10, 3)))
    print(ship.damaged_cells)
    
    ship2 = Ship(start=(4, 1), end=(4, 5))
    print(ship.is_near_ship(ship2))

    # For Task 3
    ships = ShipFactory().generate_ships()
    print(ships)