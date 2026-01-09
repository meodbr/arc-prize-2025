import random
import torch

from arc_tartiflette.graph.grid import Grid, Node
from arc_tartiflette.utils.constants import MAX_ARC_GRID_SHAPE


def get_default_arc_token_mapping(tokenizer) -> dict[str, str]:
    # Each number has a different token_id depending on the direction from which it is seen
    # For example, '1' when given as input to it's left neighbor has a different token_id than '1' when given as input to it's right neighbor
    directions = range(8)  # 8 directions
    colors = list(range(10)) + [-1]  # digits 0-9 and -1 for the wall
    original = {num: tokenizer(str(num))["input_ids"][-1] for num in range(10)}
    original[-1] = tokenizer("<wall>")["input_ids"][-1]
    direction_mapping = {
        color: [
            tokenizer(f"<{color}_{direction}>")["input_ids"][-1] 
            for direction in directions
        ]
        for color in colors
    }
    return {
        "original": original,
        "directions": direction_mapping,
        "out_of_bounds": tokenizer("<oob>")["input_ids"][-1],
    }


class ArcGrid(Grid):
    def __init__(self, list_grid: list[list[int]], token_mapping: dict, set_walls: bool=True, name="grid"):
        # Surround with walls (value=-1)
        height = len(list_grid)
        width = len(list_grid[0]) if height > 0 else 0


        new_grid_data = []
        if set_walls:
            new_grid_data = [[-1]*(width+2)]
            for row in list_grid:
                new_grid_data.append([-1]+row+[-1])
            new_grid_data.append([-1]*(width+2))
        else:
            new_grid_data = list_grid
    
        super().__init__(new_grid_data)
        self.token_mapping = token_mapping
        self.generated = False
        self.name = name
    

    @classmethod
    def for_generation(cls, token_mapping, name="grid_gen"):
        max_shape = MAX_ARC_GRID_SHAPE
        list_grid = [[-2]*(max_shape[0] + 2)] * (max_shape[1] + 2)
        print(f"list_grid for generation: {list_grid}")
        obj = cls(list_grid, token_mapping, set_walls=False, name=name)
        obj.generated = True
        return obj
    

    def _prune_out_of_bounds(self, wall_pos: list[int]):
        w_x, w_y = wall_pos

        # Don't prune if already pruned
        if w_x+1 < self.width and w_y+1 < self.height:
            if self.nodes[w_y+1][w_x+1].value == -3:
                return


        start_prune_x = w_x + 1 if w_y != 1 else 0
        start_prune_y = w_y + 1 if w_x != 1 else 0
        for y in range(start_prune_x, self.height):
            for x in range(start_prune_x, self.width):
                node = self.nodes[y][x]
                if node.value >= 0:
                    print(f"Warning: End wall placed {wall_pos}, and  found {node.value}-colored node further ({x}, {y})")
                else:
                    node.value = -3
        
        pruned_explorable_nodes = []

        for node in self.explorable_nodes:
            x, y = node.grid_position
            if x > w_x and y > w_y:
                continue
            if w_x == 1 and y > w_y:
                node.value = -3
                continue
            if w_y == 1 and x > w_x:
                node.value = -3
                continue
            pruned_explorable_nodes.append(node)
        self.explorable_nodes = pruned_explorable_nodes
    

    def assign_value(self, node: Node, value):
        assert self.generated, "Can only assign values in generated ArcGrids"
        x, y = node.grid_position
        if x > 0 and y > 0 and value == -1: # If we are assigning a non trivial wall
            self._prune_out_of_bounds([x, y])
        
        if node.value != -2 and not (x == 0 and y == 0):
            print(f"Warning, assigning {value} to node {node.grid_position}, but overriding previous value ({node.value})")
        
        node.value = value
        self.mark_node_visited(node)
    
    def assign_value_at(self, x: int, y: int, value: int):
        node = self.nodes[y][x]
        self.assign_value(node, value)


    def extract_2D_grid(self) -> list[list[int]]:
        # Get list of values without walls
        grid_2D = []
        for y in range(1, self.height-1):
            row = []
            for x in range(1, self.width-1):
                row.append(self.nodes[y][x].value)
            grid_2D.append(row)

        # Search for walls signaling a smaller grid
        for y in range(len(grid_2D)):
            x = 1
            if grid_2D[y][x] == -1:
                grid_2D = grid_2D[:y]
                break
        width = len(grid_2D[0]) if len(grid_2D) > 0 else 0
        for x in range(width):
            y = 1
            if grid_2D[y][x] == -1:
                grid_2D = [row[:x] for row in grid_2D]
                break
        
        # Validate values
        for row in grid_2D:
            for val in row:
                if val < 0 or val > 9:
                    print(f"Warning: extracted grid has invalid value {val}")
                    val = 0

        return grid_2D


    def __str__(self):
        grid_str = ""
        for row in self.nodes:
            row_str = ""
            for node in row:
                if node.value == -1:
                    row_str += "W"
                elif node.value == -2:
                    row_str += "."
                elif node.value == -3:
                    row_str += "X"
                else:
                    row_str += f"{node.value}"
            grid_str += row_str + "\n"
        return grid_str

    def _sample_node(self, nodes: list[Node], w_n=0., w_c=0.):
        weights = []
        for node in nodes:
            coef = 1.
            visited_nei = [no for no in node.get_visible_neighbors() if no is not None]
            same_col_nei = [no for no in visited_nei if no.value == node.value]
            coef += w_n*len(visited_nei)
            coef += w_c*len(same_col_nei)
            weights.append(coef)
        
        print(weights)
        selected_node = random.choices(nodes, weights, k=1)[0]
        return selected_node

    def random_exploration(self, w_n=0., w_c=0.):
        """
        Perform a random exploration of the grid starting from the start node.
        Returns a list of tokenized nodes in the order they were visited.
        """
        explorable = self.get_explorable_nodes()

        while len(explorable) > 0:
            next_node = self._sample_node(explorable, w_n=w_n, w_c=w_c)
            assert next_node and not next_node.visited, "Next node must be unvisited and valid"
            self.mark_node_visited(next_node)
            explorable = self.get_explorable_nodes()

        for row in self.nodes:
            assert all(node.visited for node in row if node.value != -1), "All non-wall nodes should be visited"

        tokenized_nodes = [node.tokenized(self.token_mapping) for node in self.visited_nodes]
        reformatted = {
            "input_ids": [tn["input_ids"] for tn in tokenized_nodes],
            "position_ids": [tn["position_ids"] for tn in tokenized_nodes],
            "labels": [tn["labels"] for tn in tokenized_nodes],
        }
        # Shifting input_ids to the left
        input_ids_to_insert_at_end = [reformatted["labels"][-1]] + [self.token_mapping["out_of_bounds"]]*7
        reformatted["input_ids"] = reformatted["input_ids"][1:] + [input_ids_to_insert_at_end]
        return {
            "input_ids": torch.tensor(reformatted["input_ids"], dtype=torch.long),
            "position_ids": torch.tensor(reformatted["position_ids"], dtype=torch.long),
            "labels": torch.tensor(reformatted["labels"], dtype=torch.long),
        }

if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meo-des/nemo_arc_main_base_1s5e_m")
    DEFAULT_ARC_TOKEN_MAPPING = get_default_arc_token_mapping(tokenizer)
    print(f"Default ARC token mapping: {DEFAULT_ARC_TOKEN_MAPPING}")
    grid_data = [
        [0, 0, 0, 1, 0],
        [1, 0, 1, 0, 0],
        [0, 0, 0, 1, 1],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 1, 0],
    ]
    grid_data = [[1]]
    arc_grid = ArcGrid(grid_data, token_mapping=DEFAULT_ARC_TOKEN_MAPPING)
    print(f"ArcGrid created with size {arc_grid.width}x{arc_grid.height}")
    print(f"grid : {arc_grid.nodes}")
    start_node = arc_grid.get_start_node()
    print(f"Start node position: {start_node.grid_position}, value: {start_node.value}")
    neighbors = start_node.get_neighbors()
    for i, neighbor in enumerate(neighbors):
        if neighbor:
            print(f"Neighbor {i} position: {neighbor.grid_position}, value: {neighbor.value}")
        else:
            print(f"Neighbor {i} is out of bounds.")
    
    arc_grid.get_explorable_nodes()
    print(arc_grid.random_exploration())