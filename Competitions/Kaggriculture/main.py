"""
Kaggriculture: Deterministic Strategic Farm Planning Agent
A multi-stage heuristic agent optimizing land expansion, crop rotation, and market timing.
"""

def agent(obs, config):
    player = obs["player"]
    step = obs["step"]
    day = obs.get("day", step // 24)
    hour = obs.get("hour", step % 24)
    
    my_farm = obs["farms"][player]
    money = my_farm["money"]
    farmer_pos = my_farm["farmer"]
    tiles = my_farm["tiles"]
    unlocked = my_farm["unlocked_quadrants"]
    hires_today = my_farm.get("hires_today", 0)
    
    private = obs.get("private", {})
    shed = private.get("shed", {})
    seeds = private.get("seeds", {})
    
    market_prices = obs.get("market", {}).get("prices", {})
    
    action = {
        "farmer": ["PASS"],
        "hands": [],
        "market": []
    }
    
    # 1. Market Orders: Liquidate harvested inventory at high price thresholds
    for item, qty in shed.items():
        if qty > 0 and item not in ["FERTILIZER"]:
            current_price = market_prices.get(item, 10)
            if current_price >= 15 or day >= 28:
                action["market"].append(["SELL", item, min(qty, 5)])
                if len(action["market"]) >= 8:
                    break

    # 2. Land Expansion: Unlock NE quadrant if capital allows
    if "NE" not in unlocked and money >= 1600:
        action["farmer"] = ["BUY_LAND", "NE"]
        return action

    # 3. Seed Purchasing: Maintain seed stock based on season progression
    if seeds.get("wheat", 0) < 3 and money >= 100:
        if day < 26:
            action["market"].append(["BUY_SEED", "wheat", 4])
    if seeds.get("carrot", 0) < 2 and money >= 200:
        if day < 26:
            action["market"].append(["BUY_SEED", "carrot", 3])
    if seeds.get("melon", 0) < 2 and money >= 600 and day <= 18:
        action["market"].append(["BUY_SEED", "melon", 2])

    # 4. Farmer Task Execution: Prioritize Harvesting > Watering > Planting
    fx, fy = farmer_pos[0], farmer_pos[1]
    tile_under = tiles[fy][fx] if 0 <= fy < len(tiles) and 0 <= fx < len(tiles[0]) else None
    
    if tile_under is not None and isinstance(tile_under, dict):
        kind = tile_under.get("kind")
        if kind == "PLANT":
            # Harvest if ready
            if tile_under.get("yield_units", 0) > 0:
                action["farmer"] = ["HARVEST"]
                return action
            # Water if not watered today
            if not tile_under.get("watered_today", False):
                action["farmer"] = ["WATER"]
                return action
        elif kind == "WEED":
            action["farmer"] = ["DIG"]
            return action

    # 5. Tile Maintenance: Scan local quadrant for actionable tiles
    for r in range(min(5, len(tiles))):
        for c in range(min(5, len(tiles[0]))):
            t = tiles[r][c]
            if t is None:
                # Empty tile: plant if seed available
                if seeds.get("wheat", 0) > 0 and day < 26:
                    if fx == c and fy == r:
                        action["farmer"] = ["PLANT", "wheat"]
                        return action
                    else:
                        dx = c - fx
                        dy = r - fy
                        if abs(dx) > abs(dy):
                            action["farmer"] = ["EAST" if dx > 0 else "WEST"]
                        else:
                            action["farmer"] = ["SOUTH" if dy > 0 else "NORTH"]
                        return action
            elif isinstance(t, dict) and t.get("kind") == "PLANT":
                if not t.get("watered_today", False) or t.get("yield_units", 0) > 0:
                    dx = c - fx
                    dy = r - fy
                    if dx != 0 or dy != 0:
                        if abs(dx) > abs(dy):
                            action["farmer"] = ["EAST" if dx > 0 else "WEST"]
                        else:
                            action["farmer"] = ["SOUTH" if dy > 0 else "NORTH"]
                        return action

    # Default idle action
    action["farmer"] = ["PASS"]
    return action