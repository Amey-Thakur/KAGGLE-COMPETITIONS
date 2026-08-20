"""
Kaggriculture: Deterministic Strategic Farm Planning Agent
Optimizing capital accumulation through crop rotations, water management, and market liquidity.
"""

def agent(obs, config=None):
    player = obs["player"]
    step = obs.get("step", 0)
    day = obs.get("day", step // 24)
    hour = obs.get("hour", step % 24)
    
    my_farm = obs["farms"][player]
    money = my_farm["money"]
    farmer_pos = my_farm["farmer"]
    tiles = my_farm["tiles"]
    unlocked = my_farm["unlocked_quadrants"]
    
    private = obs.get("private", {})
    shed = private.get("shed", {})
    seeds = private.get("seeds", {})
    
    market_prices = obs.get("market", {}).get("prices", {})
    
    action = {
        "farmer": ["PASS"],
        "hands": [],
        "market": []
    }
    
    # 1. Market Liquidation: Sell produce when prices are favorable or season is ending
    for item, qty in shed.items():
        if qty > 0 and item != "FERTILIZER":
            price = market_prices.get(item, 10)
            if price >= 15 or day >= 28:
                action["market"].append(["SELL", item, min(qty, 5)])
                if len(action["market"]) >= 8:
                    break

    # 2. Land Expansion: Buy NE quadrant once liquid reserve buffer is reached
    if "NE" not in unlocked and money >= 1600:
        action["farmer"] = ["BUY_LAND", "NE"]
        return action

    # 3. Seed Replenishment using official uppercase identifiers
    if seeds.get("WHEAT", 0) < 3 and money >= 80 and day < 26:
        action["market"].append(["BUY_SEED", "WHEAT", 4])
    if seeds.get("CARROT", 0) < 2 and money >= 160 and day < 26:
        action["market"].append(["BUY_SEED", "CARROT", 3])
    if seeds.get("MELON", 0) < 2 and money >= 600 and day <= 18:
        action["market"].append(["BUY_SEED", "MELON", 2])

    # 4. Immediate Tile Action
    fx, fy = farmer_pos[0], farmer_pos[1]
    tile = tiles[fy][fx] if 0 <= fy < len(tiles) and 0 <= fx < len(tiles[0]) else None
    
    if isinstance(tile, dict):
        kind = tile.get("kind")
        if kind == "PLANT":
            # Priority A: Harvest ripe crops
            if tile.get("yield_units", 0) > 0:
                action["farmer"] = ["HARVEST"]
                return action
            # Priority B: Water unwatered crops immediately to prevent weed conversion
            if not tile.get("watered_today", False):
                action["farmer"] = ["WATER"]
                return action
        elif kind == "WEED":
            action["farmer"] = ["DIG"]
            return action

    # 5. Local Grid Navigation & Priority Execution (NW Quadrant: 0-4, 0-4)
    # Search for pending crops needing watering or harvesting first
    for r in range(min(5, len(tiles))):
        for c in range(min(5, len(tiles[0]))):
            t = tiles[r][c]
            if isinstance(t, dict) and t.get("kind") == "PLANT":
                if not t.get("watered_today", False) or t.get("yield_units", 0) > 0:
                    dx = c - fx
                    dy = r - fy
                    if dx != 0 or dy != 0:
                        if abs(dx) > abs(dy):
                            action["farmer"] = ["EAST" if dx > 0 else "WEST"]
                        else:
                            action["farmer"] = ["SOUTH" if dy > 0 else "NORTH"]
                        return action

    # Search for empty tiles to plant
    for r in range(min(5, len(tiles))):
        for c in range(min(5, len(tiles[0]))):
            t = tiles[r][c]
            if t is None:
                # Select seed type based on day horizon
                seed_to_plant = "MELON" if (seeds.get("MELON", 0) > 0 and day <= 18) else ("CARROT" if seeds.get("CARROT", 0) > 0 else "WHEAT")
                if seeds.get(seed_to_plant, 0) > 0 and day < 26:
                    if fx == c and fy == r:
                        action["farmer"] = ["PLANT", seed_to_plant]
                        return action
                    dx = c - fx
                    dy = r - fy
                    if abs(dx) > abs(dy):
                        action["farmer"] = ["EAST" if dx > 0 else "WEST"]
                    else:
                        action["farmer"] = ["SOUTH" if dy > 0 else "NORTH"]
                    return action

    action["farmer"] = ["PASS"]
    return action