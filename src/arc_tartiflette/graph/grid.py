
class Node:
    def __init__(self, value, parent=None, grid_position=None):
        self.value = value
        self.parent = parent
        self.grid_position = grid_position
        self.visited = False
        self.visible_neighbors_when_visited = []

    def get_neighbors(self):
        neighbors = []
        x, y = self.grid_position
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]  # left, right, up, down, diagonals
        grid = self.parent
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid.width and 0 <= ny < grid.height:
                neighbors.append(grid.nodes[ny][nx])
            else:
                neighbors.append(None)
        return neighbors
    
    def get_visible_neighbors(self):
        return [
            (n if (n and n.visited) else None) 
            for n in self.get_neighbors()
        ]
    
    def get_unvisited_neighbors(self):
        return [
            (n if (n and not n.visited) else None)
            for n in self.get_neighbors() 
        ]
    
    def mark_visited(self):
        assert self.visited is False, "Node has already been visited"
        self.visited = True
        self.visible_neighbors_when_visited = self.get_visible_neighbors()
        return self.visible_neighbors_when_visited
    
    def tokenized(self, token_mapping):
        neighbors = self.visible_neighbors_when_visited
        if not self.visited:
            print("Warning: tokenizing unvisited node")
            neighbors = self.get_visible_neighbors()

        input_ids = []
        position_ids = list(self.grid_position)
        labels = token_mapping["original"][self.value]

        for direction, neighbor in enumerate(neighbors):
            if neighbor:
                input_ids.append(token_mapping["directions"][neighbor.value][direction])
            else:
                input_ids.append(token_mapping["out_of_bounds"])

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "labels": labels
        }


class Grid:
    def __init__(self, data_2d_list):
        self.height = len(data_2d_list)
        self.width = len(data_2d_list[0]) if self.height > 0 else 0
        self.nodes = [[Node(value=0, parent=self, grid_position=(x, y)) for x in range(self.width)] for y in range(self.height)]
        assert self.height == 0 or all(len(row) == len(data_2d_list[0]) for row in data_2d_list), "All rows must have the same length"

        for y in range(self.height):
            for x in range(self.width):
                self.nodes[y][x].value = data_2d_list[y][x]

    def get_explorable_nodes(self):
        explorable = []
        for row in self.nodes:
            for node in row:
                if node.visited:
                    for neighbor in node.get_unvisited_neighbors():
                        if neighbor is not None:
                            explorable.append(neighbor)
        # Deduplicate based on grid_position
        explorable = list({node.grid_position: node for node in explorable}.values())
        return explorable