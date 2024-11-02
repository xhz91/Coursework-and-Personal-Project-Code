from network.graph import NeighbourGraphBuilder
import heapq

class PathFinder:
    """
    Task 3: Complete the definition of the PathFinder class by:
    - completing the definition of the __init__() method (if needed)
    - completing the "get_shortest_path()" method (don't hesitate to divide 
      your code into several sub-methods)
    """

    def __init__(self, tubemap):
        """
        Args:
            tubemap (TubeMap) : The TubeMap to use.
        """
        self.tubemap = tubemap

        graph_builder = NeighbourGraphBuilder()
        self.graph = graph_builder.build(self.tubemap)
        
        # Feel free to add anything else needed here.


    def get_station_id_from_name(self, station_name):
        """ Sub-method used to find the station id from the given station name.

        Args:
            station_name (str): name of the station

        Returns:
            station.id (str): id of the station
        """
        for station in self.tubemap.stations.values():
            if station.name == station_name:
                return station.id
        raise ValueError(f"Station with name '{station_name}' does not exist.")
            

    def initialise_distance(self, start_station_id):
        """ Sub-method used to initialise distance to source for each station.
            Distance for start station is set to 0, and inf for all other stations.

        Args:
            start_station_id (str): id of the start station
        
        Returns:
            distance (dict): a dictionary of distances from a station to the source station.
                            the keys are station ids, and values are the distance from the source station.
        """
        distance = {station_id: float("inf") for station_id in self.graph}
        distance[start_station_id] = 0
        return distance
    

    def initialise_previous(self):
        """ Sub-method used to initialise previous station. This is set to None for all stations
        
        Returns:
            a dictionary prev with keys - station ids, and values - None for all stations
        """
        prev = {station_id: None for station_id in self.graph}
        return prev


    def reconstruct_path(self, prev, end_station_id):
        """ Sub-method used to reconstruct the shortest path, which is a list containing station names on
            the shortest path
        
        Args:
            prev (dict): a dictionary that stores the previous node for each station
            end_station_id (str): id of the target destination station
        
        Returns:
            shortest_path (list): list containing all the station names on the shortest path
        """
        shortest_path_with_id = []
        current_station_id = end_station_id

        while current_station_id is not None:
            shortest_path_with_id.insert(0,current_station_id)
            current_station_id = prev[current_station_id]
        
        shortest_path = []
        for station_id in shortest_path_with_id:
            shortest_path.append(self.tubemap.stations[station_id])
        return shortest_path


    def get_shortest_path(self, start_station_name, end_station_name):
        """ Find ONE shortest path from start_station_name to end_station_name.
        
        The shortest path is the path that takes the least amount of time.

        For instance, get_shortest_path('Stockwell', 'South Kensington') 
        should return the list:
        [Station(245, Stockwell, {2}), 
         Station(272, Vauxhall, {1, 2}), 
         Station(198, Pimlico, {1}), 
         Station(273, Victoria, {1}), 
         Station(229, Sloane Square, {1}), 
         Station(236, South Kensington, {1})
        ]

        If start_station_name or end_station_name does not exist, return None.
        
        You can use the Dijkstra algorithm to find the shortest path from
        start_station_name to end_station_name.

        Find a tutorial on YouTube to understand how the algorithm works, 
        e.g. https://www.youtube.com/watch?v=GazC3A4OQTE
        
        Alternatively, find the pseudocode on Wikipedia: https://en.wikipedia.org/wiki/Dijkstra's_algorithm#Pseudocode

        Args:
            start_station_name (str): name of the starting station
            end_station_name (str): name of the ending station

        Returns:
            list[Station] : list of Station objects corresponding to ONE 
                shortest path from start_station_name to end_station_name.
                Returns None if start_station_name or end_station_name does not 
                exist.
                Returns a list with one Station object (the station itself) if 
                start_station_name and end_station_name are the same.
        """
        # Identify the id of the start station and end station, if the name input is invalid, return None
        try:
            start_station_id = self.get_station_id_from_name(start_station_name)
            end_station_id = self.get_station_id_from_name(end_station_name)
        except ValueError as error:
            print(error)
            return None
        
        # Return a list with an instance of the station itself if start and end station is the same
        if start_station_id == end_station_id:
            return [self.tubemap.stations[start_station_id]]
        
        # If inputs are valid, continue with the Dijkstra algorithm
        # Initialise distance and previous nodes
        distance = self.initialise_distance(start_station_id)
        prev = self.initialise_previous()

        # Define a vertex priority queue, which stores distance of station to start station and station id in a tuple
        priority_queue = []
        for station_id in self.graph:
            priority_queue.append((distance[station_id],station_id))
        
        # Use heapify to rearrange the priority_queue list in to a heap
        heapq.heapify(priority_queue)

        while priority_queue:
            # Extract the station with minimum distance from the start station
            current_distance, current_station_id = heapq.heappop(priority_queue)
            
            # Exit the loop if we have found the end station
            if current_station_id == end_station_id:
                break
            
            # Iterate through all neighbours of extracted station and all connection instances between them
            for neighbour_id, connections in self.graph[current_station_id].items():
                for connection in connections:
                    new_distance = current_distance + connection.time

                    # If new distance is smaller than currently know distance to the neighbour: 
                    # update the distance and set current station to be the previous of neighbour station
                    if new_distance < distance[neighbour_id]:
                        distance[neighbour_id] = new_distance
                        prev[neighbour_id] = current_station_id
                        
                        # Push the neighbour station with its updated distance to priority queue, so that priority queue alwas has shortest path
                        heapq.heappush(priority_queue,(new_distance,neighbour_id))
        
        shortest_path = self.reconstruct_path(prev, end_station_id)
        return shortest_path


def test_shortest_path():
    from tube.map import TubeMap
    tubemap = TubeMap()
    tubemap.import_from_json("data/london.json")
 
    path_finder = PathFinder(tubemap)
    stations = path_finder.get_shortest_path("Covent Garden", "Green Park")
    print(stations)
    
    station_names = [station.name for station in stations]
    expected = ["Covent Garden", "Leicester Square", "Piccadilly Circus", 
                "Green Park"]
    assert station_names == expected


if __name__ == "__main__":
    test_shortest_path()
