class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        x = init
        for _ in range(iterations):
            gradient = 2 * x
            x = x - learning_rate * gradient
        
        # Handle the case where no iterations occur to return exactly the input type/value
        if iterations == 0:
            return init
            
        return round(float(x), 5)