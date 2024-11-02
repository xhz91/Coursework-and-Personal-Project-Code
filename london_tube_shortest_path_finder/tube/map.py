import json
from tube.components import Station, Line, Connection

class TubeMap:
    """
    Task 1: Complete the definition of the TubeMap class by:
    - completing the "import_from_json()" method

    Don't hesitate to divide your code into several sub-methods, if needed.

    As a minimum, the TubeMap class must contain these three member attributes:
    - stations: a dictionary that indexes Station instances by their id 
      (key=id (str), value=Station)
    - lines: a dictionary that indexes Line instances by their id 
      (key=id, value=Line)
    - connections: a list of Connection instances for the TubeMap 
      (list of Connections)
    """

    def __init__(self):
        self.stations = {}  # key: id (str), value: Station instance
        self.lines = {}  # key: id (str), value: Line instance
        self.connections = []  # list of Connection instances

    def convert_zone(self, zone_string):
        """ Sub-method for converting the zone for Station instances into set of integers
            
            Args:
            zone_string (str): value of zone in string type, e.g. "1", "1.5"
            
            Returns:
            set of integers, e.g. {1} for "1", {1,2} for "1.5"

        """
        if "." not in zone_string:
            return {int(zone_string)}
        else:
            zone_float = float(zone_string)
            lower_int = int(zone_float)
            upper_int = lower_int + 1
            return {lower_int,upper_int}


    def import_from_json(self, filepath):
        """ Import tube map information from a JSON file.
        
        During the import process, the `stations`, `lines` and `connections` 
        attributes should be updated.

        You can use the `json` python package to easily load the JSON file at 
        `filepath`

        Note: when the indicated zone is not an integer (for instance: "2.5"), 
            it means that the station belongs to two zones. 
            For example, if the zone of a station is "2.5", 
            it means that the station is in both zones 2 and 3.

        Args:
            filepath (str) : relative or absolute path to the JSON file 
                containing all the information about the tube map graph to 
                import. If filepath is invalid, no attribute should be updated, 
                and no error should be raised.

        Returns:
            None
        """
        # Load in data from json file
        with open(filepath, "r") as jsonfile:
            data = json.load(jsonfile)

            # Update the stations (dictionary) attribute
            for station_item in data["stations"]:
                # Extract the id of stations
                key_stations = station_item["id"]
                
                # Concert str type zone from imported data into set of integers
                zones = self.convert_zone(station_item["zone"])
                
                # Construct the stations attribute of TubeMap
                self.stations[key_stations] = Station(id=station_item["id"],
                                                      name=station_item["name"],
                                                      zones=zones)

            # Update the lines (dictionary) attribute
            for line_item in data["lines"]:
                # Extract the id of lines
                key_lines = line_item["line"]
                
                # Construct the lines attribute of TubeMap
                self.lines[key_lines] = Line(id=line_item["line"],
                                             name=line_item["name"])

            # Update the connections attribute
            for connection_item in data["connections"]:
                # Retrive the Station and Line instances from self.stations and self.lines using .get()
                station1 = self.stations.get(connection_item["station1"])
                station2 = self.stations.get(connection_item["station2"])
                line = self.lines.get(connection_item["line"])

                # Construct the indivisual connection instance of the connections attribute of TubeMap
                connection_instance = Connection(stations = {station1,station2},
                                        line=line,
                                        time=int(connection_item["time"]))

                # Construct the connections (list) attribute of TubeMap
                self.connections.append(connection_instance)
            
        return None
    

def test_import():
    tubemap = TubeMap()
    tubemap.import_from_json("data/london.json")
    
    # view one example Station
    print(tubemap.stations[list(tubemap.stations)[0]])

    # view one example Line
    print(tubemap.lines[list(tubemap.lines)[0]])
    
    # view the first Connection
    print(tubemap.connections[0])

    # view stations for the first Connection
    print([station for station in tubemap.connections[0].stations])


if __name__ == "__main__":
    test_import()
