import numpy as np

class MathLevelEngine:
    def __init__(self, mode="Daily"):
        self.mode = mode
        self.mult = 4 if mode == "Daily" else 8 if mode == "Weekly" else 12
        
        self.period_open = None
        self.price_diff = None
        
        self.upper_levels = np.full(12, np.nan)
        self.lower_levels = np.full(12, np.nan)
        self.mid_upper_levels = np.full(12, np.nan)
        self.mid_lower_levels = np.full(12, np.nan)
        
    def reset_levels(self, current_open):
        self.period_open = current_open
        
        scale_factor = 1000.0
        calc_open = self.period_open * scale_factor
        sqrt_open = np.sqrt(calc_open)
        sqrt_trunc = np.floor(sqrt_open)
        sqrt_plus_two = sqrt_trunc + 2
        squared_result = sqrt_plus_two * sqrt_plus_two
        diff = squared_result - calc_open
        final_diff = diff * self.mult
        self.price_diff = final_diff / scale_factor
        
        for i in range(12):
            self.upper_levels[i] = self.period_open + self.price_diff * (i + 1)
            self.lower_levels[i] = self.period_open - self.price_diff * (i + 1)
            
        # mid-levels
        self.mid_upper_levels[0] = (self.period_open + self.upper_levels[0]) / 2
        self.mid_lower_levels[0] = (self.period_open + self.lower_levels[0]) / 2
        
        for i in range(1, 12):
            self.mid_upper_levels[i] = (self.upper_levels[i-1] + self.upper_levels[i]) / 2
            self.mid_lower_levels[i] = (self.lower_levels[i-1] + self.lower_levels[i]) / 2
            
    def get_all_levels(self):
        """Returns a sorted list of all active levels with their identifiers dicts."""
        if self.period_open is None:
            return []
            
        levels = []
        levels.append({"price": self.period_open, "id": "Open", "type": "Open", "rank": 0})
        
        for i in range(12):
            cat = "Base" if i < 4 else "Extended" if i < 8 else "SuperExtended"
            rank = i + 1
            levels.append({"price": self.upper_levels[i], "id": f"U{i+1}", "type": cat, "rank": rank})
            levels.append({"price": self.lower_levels[i], "id": f"L{i+1}", "type": cat, "rank": -rank}) # -rank for lowers visually
            
            levels.append({"price": self.mid_upper_levels[i], "id": f"Mid_U{i+1}", "type": f"Mid_{cat}", "rank": rank - 0.5})
            levels.append({"price": self.mid_lower_levels[i], "id": f"Mid_L{i+1}", "type": f"Mid_{cat}", "rank": -(rank - 0.5)})
            
        return sorted(levels, key=lambda x: x["price"])
        
    def find_nearest_context(self, price):
        levels = self.get_all_levels()
        if not levels:
            # Default fallback if no session started
            return {
                "nearest_price": price,
                "dist_to_level": 0.0,
                "nearest_id": "None",
                "nearest_type": "None",
                "dist_prev_level": 0.0,
                "dist_next_level": 0.0,
                "level_position": 0.5,
                "nearest_rank": 0
            }
            
        prices = [lvl["price"] for lvl in levels]
        
        # Find nearest
        nearest_idx = np.argmin([abs(price - p) for p in prices])
        nearest = levels[nearest_idx]
        
        dist_to_nearest = abs(price - nearest["price"])
        
        prev_level = None
        next_level = None
        
        for i in range(len(prices)-1):
            if prices[i] <= price <= prices[i+1]:
                prev_level = prices[i]
                next_level = prices[i+1]
                break
                
        if prev_level is None:
            if price < prices[0]:
                prev_level = prices[0] - self.price_diff
                next_level = prices[0]
            else:
                prev_level = prices[-1]
                next_level = prices[-1] + self.price_diff
                
        dist_prev = abs(price - prev_level)
        dist_next = abs(next_level - price)
        gap = next_level - prev_level
        position = (price - prev_level) / gap if gap > 0 else 0
        
        return {
            "nearest_price": nearest["price"],
            "dist_to_level": dist_to_nearest,
            "nearest_id": nearest["id"],
            "nearest_type": nearest["type"],
            "dist_prev_level": dist_prev,
            "dist_next_level": dist_next,
            "level_position": position,
            "nearest_rank": nearest.get("rank", 0)
        }