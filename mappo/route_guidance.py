from collections import deque

from environment.obstacles import Point
from environment.movement import MOVES, RECHARGE, WAIT


def _shortest_first_actions(city, start, goal, occupied):
    if start == goal:
        return [], 0

    queue = deque([start])
    distances = {start: 0}
    parents = {}

    while queue:
        current = queue.popleft()
        if current == goal:
            break

        for action, (dx, dy) in MOVES.items():
            neighbor = Point(current.x + dx, current.y + dy)
            if (
                neighbor in distances
                or neighbor in occupied
                or not city.inside_city(neighbor.x, neighbor.y)
                or city.is_blocked(neighbor.x, neighbor.y)
            ):
                continue
            distances[neighbor] = distances[current] + 1
            parents[neighbor] = (current, action)
            queue.append(neighbor)

    if goal not in distances:
        return None, None

    actions = []
    current = goal
    while current != start:
        previous, action = parents[current]
        actions.append(action)
        current = previous
    actions.reverse()
    return actions, distances[goal]


def route_guidance(env):
    """Return each drone's next shortest-path action, respecting fleet occupancy."""
    guided_actions = []
    city = env.city
    movement_cost = env.config["movement_battery_cost"]

    for drone in env.drones:
        if not drone.active:
            guided_actions.append(WAIT)
            continue

        occupied = {
            other.position
            for other in env.drones
            if other is not drone
        }
        route, distance = _shortest_first_actions(
            city, drone.position, drone.destination, occupied
        )

        if route is None:
            route, distance = _shortest_first_actions(
                city, drone.position, drone.destination, set()
            )

        if route is None:
            guided_actions.append(WAIT)
            continue

        target = drone.destination
        if distance * movement_cost > drone.battery:
            charger_routes = []
            for charger in city.charging_stations:
                to_charger, charger_distance = _shortest_first_actions(
                    city, drone.position, charger, occupied
                )
                if to_charger is None:
                    continue
                from_charger, destination_distance = _shortest_first_actions(
                    city, charger, drone.destination, set()
                )
                if (
                    from_charger is not None
                    and charger_distance * movement_cost <= drone.battery
                ):
                    charger_routes.append(
                        (charger_distance + destination_distance, charger, to_charger)
                    )

            if charger_routes:
                _, target, route = min(charger_routes, key=lambda item: item[0])
                if drone.position == target:
                    guided_actions.append(RECHARGE)
                    continue

        if not route:
            guided_actions.append(WAIT)
            continue

        guided_actions.append(route[0])

    return guided_actions


def choose_guided_actions(env, action_probabilities, action_masks):
    """Use the CNN to rank legal actions, constrained to a shortest route."""
    planned_actions = route_guidance(env)
    actions = []
    for i, planned in enumerate(planned_actions):
        if action_masks[i, planned]:
            actions.append(planned)
            continue

        legal_actions = [
            action for action, legal in enumerate(action_masks[i]) if legal
        ]
        if not legal_actions:
            raise RuntimeError(f"Drone {i} has no legal action")
        actions.append(
            max(legal_actions, key=lambda action: action_probabilities[i, action])
        )
    return actions
