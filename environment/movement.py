MOVES = {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0)}
WAIT, RECHARGE = 4, 5
ACTION_NAMES = ["up", "down", "left", "right", "wait", "recharge"]


def legal_action_mask(drone, city, config, drones=None):
    if not drone.active:
        return [False, False, False, False, True, False]

    x, y = drone.position.x, drone.position.y
    occupied = {
        (other.position.x, other.position.y)
        for other in (drones or [])
        if other is not drone
    }
    mask = [
        city.inside_city(x + dx, y + dy)
        and not city.is_blocked(x + dx, y + dy)
        and (x + dx, y + dy) not in occupied
        and drone.battery >= config["movement_battery_cost"]
        for dx, dy in MOVES.values()
    ]
    
    mask.append(drone.battery >= config["wait_battery_cost"] or config["wait_battery_cost"] == 0)
    mask.append(city.is_charging_station(x, y) and drone.battery < drone.initial_battery)

    if not any(mask):
        mask[WAIT] = True

    return mask


def execute_action(drone, action, city, config, drones=None, action_mask=None):
    if not drone.active:
        return {"moved": False, "illegal": False, "no_fly": False}

    mask = action_mask if action_mask is not None else legal_action_mask(
        drone, city, config, drones
    )
    if not (0 <= action < 6) or not mask[action]:
        return {"moved": False, "illegal": True, "no_fly": False}

    if action == WAIT:
        drone.consume_battery(config["wait_battery_cost"])
    elif action == RECHARGE:
        drone.recharge(config["recharge_amount"])
    else:
        dx, dy = MOVES[action]
        drone.consume_battery(config["movement_battery_cost"])
        drone.move_to(type(drone.position)(drone.position.x + dx, drone.position.y + dy))

    return {"moved": action < 4, "illegal": False, "no_fly": False}