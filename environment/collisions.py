def detect_collisions(previous_positions, new_positions, active_mask):
    n = len(new_positions)
    collisions = [False] * n

    for i in range(n):
        if not active_mask[i]:
            continue
        for j in range(i + 1, n):
            if not active_mask[j]:
                continue
            
            same_cell = new_positions[i] == new_positions[j]
            edge_swap = (
                previous_positions[i] == new_positions[j]
                and previous_positions[j] == new_positions[i]
                and previous_positions[i] != previous_positions[j]
            )

            if same_cell or edge_swap:
                collisions[i] = collisions[j] = True

    return collisions