import pandas as pd


def optimize(stockout_func, target=0.95, nsim=100, tol = 0.025, show: bool = False) -> np.ndarray:
    """Optimizes `in-stock rate`, the inverse of stockout rate. 
    Finds the least amount of stock that achieves the target instock rate via binary search 
    """
    def _distance_within_tol(actual, target, tol):
        distance = actual - target
        distacnce_not_above = distance <= tol
        distance_not_below = distance >= -tol
        return distacnce_not_above and distance_not_below

    def _get_init_guess(target):
        """Quickly tries to guess an upper bound estimate"""
        for guess in [500, 1000, 2500]:
            in_stock = 1 - stockout_func(guess)
            if in_stock >= target:
                return guess
        
        return 5000 # arbitrary, but never going to buy more than this

    
    last_guess, guess, in_stock = None, 50, 0
    low, high = 0, _get_init_guess(target)

    tracker = []
    while not _distance_within_tol(in_stock, target, tol) and (guess != last_guess):
        # Simulate in-stock rate for current guess
        in_stock = 1 - stockout_func(guess)
        tracker.append([guess, in_stock])

        # if the current guess is sufficient, finish
        if _distance_within_tol(in_stock, target, tol):
            break
    
        # if the current guess underpredicts stock levels...
        elif in_stock < target:
            # guess half-way between lowest point and middle point
            low = max(low, guess)
            last_guess, guess = guess, low + (high - low) // 2
    
        # if the current guess overpredict stock levels...
        elif in_stock >= target:
    
            # if we ended up guessing the same stock levels again and we're above our target, finish
            if guess == last_guess:
                break
    
            # otherwise guess half-way between highest point and middle point
            high = min(high, guess)
            last_guess, guess = guess, high - (high - low) // 2
    
        else:
            raise ValueError("Issue: unexpected in-stock rate")
    
        if show:
            print(f"guess: {last_guess}, in-stock rate: {round(in_stock,2)}")

    return pd.DataFrame(tracker, columns=['order_units','in_stock'])
    
        