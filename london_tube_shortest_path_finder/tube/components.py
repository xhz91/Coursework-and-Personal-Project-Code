"""
WARNING: the following classes should remain unchanged!
"""

class Station:
    def __init__(self, id, name, zones):
        """ A class representing a Tube station.
        
        Args:
            id (str) : Station ID
            name (str) : Station name
            zones (set[int]) : Set of zone numbers for station. Some stations
                               may belong to more than one zone.
        """
        self.id = id
        self.name = name
        self.zones = zones

    def __repr__(self):
        return f"Station({self.id}, {self.name}, {self.zones})"


class Line:
    def __init__(self, id, name):
        """ A class representing a Tube line.
        
        Args:
            id (str) : Line ID
            name (str) : Line name
        """
        self.id = id
        self.name = name

    def __repr__(self):
        return f"Line({self.id}, {self.name})"


class Connection:
    def __init__(self, stations, line, time):
        """ A connection between two stations on a specific Tube line.
        
        Args:
            stations (set[Station]) : stations associated with the connection
            line (Line) : the line for the connection
            time (int) : time needed (in minutes) to transit between the 
                         stations
        """
        self.stations = stations
        self.line = line
        self.time = time

    def __repr__(self):
        station_names = sorted([station.name for station in self.stations])
        return (f"Connection("
            f"{'<->'.join(station_names)}, {self.line.name}, {self.time})"
        )


if __name__ == '__main__':
    # Two Station instances
    station_1 = Station(id="99",
                        name="Gloucester Road",
                        zones={1})

    station_2 = Station(id="74",
                        name="Earl's Court",
                        zones={1, 2})

    # Two Line instances
    line_1 = Line(id="10",
                  name="Piccadilly Line")
                  
    line_2 = Line(id="4",
              name="District Line")

    # Two Connection instances. 
    # Note how the same set of stations can be connected by different lines.
    connection_1 = Connection(stations={station_1, station_2},
                            line=line_1,
                            time=3)
                            
    connection_2 = Connection(stations={station_1, station_2},
                            line=line_2,
                            time=4)
                            
