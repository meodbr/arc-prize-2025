
class Node:
    def __init__(self, value, parent=None, grid_position=None):
        self.value = value
        self.parent = parent
        self.grid_position = grid_position
        self.visited = False
        self.visible_neighbors_when_visited = []
        self.explorable = False

    def get_neighbors(self):
        neighbors = []
        x, y = self.grid_position
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]  # left, right, up, down, diagonals
        grid = self.parent
        print(f"Getting neighbors for node at position {self.grid_position} with value {self.value}")
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            print(f"  Checking neighbor at position {nx}, {ny}, value: ", end="")
            if 0 <= nx < grid.width and 0 <= ny < grid.height:
                neighbor = grid.nodes[ny][nx]
                print(f"{neighbor.value}")
                if neighbor.value != -3:
                    neighbors.append(neighbor)
                else:
                    neighbors.append(None)
            else:
                print("Out of bounds")
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
        self.explorable = False
        self.visible_neighbors_when_visited = self.get_visible_neighbors()
        return self.visible_neighbors_when_visited
    
    def tokenized(self, token_mapping):
        neighbors = self.visible_neighbors_when_visited
        if not self.visited:
            if not self.parent.generated:
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
        self.visited_nodes = []
        self.explorable_nodes = [self.get_start_node()]
        assert self.height == 0 or all(len(row) == len(data_2d_list[0]) for row in data_2d_list), "All rows must have the same length"

        for y in range(self.height):
            for x in range(self.width):
                self.nodes[y][x].value = data_2d_list[y][x]

    def get_start_node(self):
        return self.nodes[0][0]  # Assuming start is always at (1, 1) after adding walls

    def get_explorable_nodes(self):
        return self.explorable_nodes
    
    def mark_node_visited(self, node: Node):
        node.mark_visited()
        self.visited_nodes.append(node)
        self.explorable_nodes.remove(node)

        # Update explorable nodes
        for neighbor in node.get_unvisited_neighbors():
            print("Checking neighbor for explorable:", end=" ")
            if neighbor:
                print(neighbor.grid_position, neighbor.value)
                print("  Explorable before:", neighbor.explorable)
                if not neighbor.explorable:
                    print("  Marking as explorable")
                    neighbor.explorable = True
                    self.explorable_nodes.append(neighbor)
                else:
                    print("  Already explorable")
            else:
                print("None neighbor")
