import random
import torch

from arc_tartiflette.graph.grid import Grid, Node


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
    def __init__(self, list_grid: list[list[int]], token_mapping: dict):
        # Surround with walls (value=-1)
        height = len(list_grid)
        width = len(list_grid[0]) if height > 0 else 0


        new_grid_data = [[-1]*(width+2)]
        for row in list_grid:
            new_grid_data.append([-1]+row+[-1])
        new_grid_data.append([-1]*(width+2))
    
        super().__init__(new_grid_data)
        self.token_mapping = token_mapping

    def get_start_node(self):
        return self.nodes[0][0]  # Assuming start is always at (1, 1) after adding walls
    
    def random_exploration(self):
        """
        Perform a random exploration of the grid starting from the start node.
        Returns a list of tokenized nodes in the order they were visited.
        """
        start_node = self.get_start_node()
        current_node = start_node
        visited_nodes = [current_node]
        current_node.mark_visited()
        explorable = self.get_explorable_nodes()

        while len(explorable) > 0:
            next_node = random.choice(explorable)
            assert next_node and not next_node.visited, "Next node must be unvisited and valid"
            visited_nodes.append(next_node)
            next_node.mark_visited()
            explorable = self.get_explorable_nodes()
            print(f"Visited node at {next_node.grid_position} with value {next_node.value}. {len(explorable)} explorable nodes remaining.")
            print(f"  Visible neighbors when visited: {[n.grid_position if n else None for n in next_node.visible_neighbors_when_visited]}")

        for row in self.nodes:
            assert all(node.visited for node in row if node.value != -1), "All non-wall nodes should be visited"

        tokenized_nodes = [node.tokenized(self.token_mapping) for node in visited_nodes]
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

from transformers import AutoTokenizer
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("meo-des/nemo_arc_main_base_1s10e_m")
    DEFAULT_ARC_TOKEN_MAPPING = get_default_arc_token_mapping(tokenizer)
    print(f"Default ARC token mapping:", DEFAULT_ARC_TOKEN_MAPPING)
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