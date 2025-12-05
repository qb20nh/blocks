import numpy as np
import time
import pickle
import os
import signal
import sys
import numba
from numba import jit as njit

@njit(nopython=True)
def oklch_to_oklab(oklch):
    """Convert OKLCH to OKLAB."""
    l = oklch[:, 0]
    c = oklch[:, 1]
    h_rad = np.radians(oklch[:, 2])
    
    a = c * np.cos(h_rad)
    b = c * np.sin(h_rad)
    
    return np.stack((l, a, b), axis=1)

@njit(nopython=True)
def oklab_to_oklch(oklab):
    """Convert OKLAB to OKLCH."""
    l = oklab[:, 0]
    a = oklab[:, 1]
    b = oklab[:, 2]
    
    c = np.sqrt(a**2 + b**2)
    h_rad = np.arctan2(b, a)
    h_deg = np.degrees(h_rad) % 360
    
    return np.stack((l, c, h_deg), axis=1)

@numba.jit(nopython=True)
def oklab_to_linear_srgb(oklab):
    """Convert OKLAB to Linear sRGB."""
    l_ = oklab[:, 0] + 0.3963377774 * oklab[:, 1] + 0.2158037573 * oklab[:, 2]
    m_ = oklab[:, 0] - 0.1055613458 * oklab[:, 1] - 0.0638541728 * oklab[:, 2]
    s_ = oklab[:, 0] - 0.0894841775 * oklab[:, 1] - 1.2914855480 * oklab[:, 2]
    
    l = l_**3
    m = m_**3
    s = s_**3
    
    r = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    b = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    
    return np.stack((r, g, b), axis=1)

@numba.jit(nopython=True)
def linear_srgb_to_srgb(linear):
    """Convert Linear sRGB to sRGB (gamma correction)."""
    srgb = np.empty_like(linear)
    for i in range(linear.shape[0]):
        for j in range(linear.shape[1]):
            val = linear[i, j]
            if val <= 0.0031308:
                srgb[i, j] = 12.92 * val
            else:
                srgb[i, j] = 1.055 * (max(val, 0) ** (1/2.4)) - 0.055
    return srgb

@numba.jit(nopython=True)
def is_in_gamut(oklch):
    """Check if OKLCH colors are within sRGB gamut."""
    oklab = oklch_to_oklab(oklch)
    linear = oklab_to_linear_srgb(oklab)
    epsilon = 1e-6
    
    # Numba doesn't support axis argument in all functions efficiently, manual loop or flat check
    # But np.all with axis is supported in recent numba. Let's try simple loop for safety and speed in nopython
    n = linear.shape[0]
    result = np.empty(n, dtype=np.bool_)
    for i in range(n):
        in_g = True
        for j in range(3):
            if linear[i, j] < -epsilon or linear[i, j] > 1.0 + epsilon:
                in_g = False
                break
        result[i] = in_g
    return result

@numba.jit(nopython=True)
def constrain_to_gamut(oklch):
    """Constrain colors to sRGB gamut and Lightness range."""
    # 1. Clamp Lightness to valid sRGB range
    # oklch[:, 0] = np.clip(oklch[:, 0], 0.0, 1.0) # Numba supports clip
    
    for i in range(oklch.shape[0]):
        if oklch[i, 0] < 0.0: oklch[i, 0] = 0.0
        if oklch[i, 0] > 1.0: oklch[i, 0] = 1.0
    
    # 2. Constrain to sRGB Gamut (reduce Chroma)
    # We need to check gamut. 
    # Since we modify oklch in place, we can iterate
    
    for i in range(oklch.shape[0]):
        # Check if in gamut
        # We need a single point check, but our is_in_gamut takes array.
        # Let's make a small array or refactor is_in_gamut to take single point?
        # For simplicity, pass slice which is array view
        
        # Actually, let's just implement the logic here to avoid overhead
        # or call the array version with 1-element slice
        
        # Optimization: check if already in gamut
        # Create a 1-element array wrapper for the current color to pass to is_in_gamut
        # This might be slow inside a loop. 
        # Better: is_in_gamut logic inline or helper for single color.
        
        # Let's rely on the fact that we can pass a slice
        current_color = oklch[i:i+1]
        if is_in_gamut(current_color)[0]:
            continue
            
        l = oklch[i, 0]
        c = oklch[i, 1]
        h = oklch[i, 2]
        
        low = 0.0
        high = c
        
        # Binary search
        for _ in range(10):
            mid = (low + high) / 2
            # Construct temp array for check
            # Numba doesn't like complex array construction in loop sometimes
            # We can just modify current_color temporarily? No, it's a view.
            
            # Let's make a temp array
            temp_arr = np.empty((1, 3), dtype=np.float64)
            temp_arr[0, 0] = l
            temp_arr[0, 1] = mid
            temp_arr[0, 2] = h
            
            if is_in_gamut(temp_arr)[0]:
                low = mid
            else:
                high = mid
        
        oklch[i, 1] = low
        
    return oklch

@numba.jit(nopython=True)
def calculate_metrics(oklch):
    """Calculate min distance and std dev of NN distances."""
    oklab = oklch_to_oklab(oklch)
    n = oklab.shape[0]
    
    # Calculate distance matrix
    # Numba friendly double loop
    dist_sq = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            d = 0.0
            for k in range(3):
                diff = oklab[i, k] - oklab[j, k]
                d += diff * diff
            dist_sq[i, j] = d
            
    # Fill diagonal with inf
    for i in range(n):
        dist_sq[i, i] = np.inf
        
    dist = np.sqrt(dist_sq)
    
    min_dist = np.min(dist)
    
    min_dists = np.empty(n, dtype=np.float64)
    for i in range(n):
        min_val = np.inf
        for j in range(n):
            if dist[i, j] < min_val:
                min_val = dist[i, j]
        min_dists[i] = min_val
        
    std_dev = np.std(min_dists)
    
    return min_dist, std_dev

@numba.jit(nopython=True)
def calculate_score(min_dist, std_dev):
    return min_dist - 1.0 * std_dev

class ColorOptimizer:
    def __init__(self, num_colors=27):
        self.num_colors = num_colors
        self.rng = np.random.default_rng(42)
        
        # Simulated Annealing parameters
        self.initial_temp = 1.0
        self.final_temp = 0.0001
        self.iterations = 100000000
        self.state_file = "optimizer_state.pkl"
        self.start_iteration = 0
        
        # Initialize random colors in OKLCH
        self.colors_oklch = np.zeros((num_colors, 3))
        # Start with a random distribution
        self.colors_oklch[:, 0] = self.rng.uniform(0.0, 1.0, num_colors)
        self.colors_oklch[:, 1] = self.rng.uniform(0.0, 0.5, num_colors)
        self.colors_oklch[:, 2] = self.rng.uniform(0, 360, num_colors)

    def oklch_to_oklab(self, oklch):
        return oklch_to_oklab(oklch)

    def oklab_to_oklch(self, oklab):
        return oklab_to_oklch(oklab)

    def oklab_to_linear_srgb(self, oklab):
        return oklab_to_linear_srgb(oklab)

    def linear_srgb_to_srgb(self, linear):
        return linear_srgb_to_srgb(linear)

    def is_in_gamut(self, oklch):
        return is_in_gamut(oklch)

    def constrain_to_gamut(self, oklch):
        return constrain_to_gamut(oklch)

    def calculate_metrics(self, oklch):
        return calculate_metrics(oklch)

    def calculate_score(self, min_dist, std_dev):
        return calculate_score(min_dist, std_dev)

    def save_state(self, iteration, current_score, current_min_dist, current_std_dev, best_score, best_colors):
        """Save the current state to a file."""
        print(f"\nSaving state at iteration {iteration}...")
        state = {
            'iteration': iteration,
            'colors_oklch': self.colors_oklch,
            'current_score': current_score,
            'current_min_dist': current_min_dist,
            'current_std_dev': current_std_dev,
            'best_score': best_score,
            'best_colors': best_colors,
            'rng_state': self.rng.bit_generator.state
        }
        with open(self.state_file, 'wb') as f:
            pickle.dump(state, f)
        print("State saved.")

    def load_state(self):
        """Load state from file if it exists."""
        if os.path.exists(self.state_file):
            print(f"Found saved state file: {self.state_file}")
            try:
                with open(self.state_file, 'rb') as f:
                    state = pickle.load(f)
                
                self.start_iteration = state['iteration']
                self.colors_oklch = state['colors_oklch']
                self.rng.bit_generator.state = state['rng_state']
                
                print(f"Resuming from iteration {self.start_iteration}")
                return True, state
            except Exception as e:
                print(f"Failed to load state: {e}")
                return False, None
        return False, None

    def optimize(self):
        print(f"Optimizing {self.num_colors} colors using Hybrid Simulated Annealing...")
        
        # Try to load state
        loaded, state = self.load_state()
        
        if loaded:
            current_score = state['current_score']
            current_min_dist = state['current_min_dist']
            current_std_dev = state['current_std_dev']
            best_score = state['best_score']
            best_colors = state['best_colors']
        else:
            # Ensure valid start
            self.colors_oklch = self.constrain_to_gamut(self.colors_oklch)
            
            current_min_dist, current_std_dev = self.calculate_metrics(self.colors_oklch)
            current_score = self.calculate_score(current_min_dist, current_std_dev)
            
            best_score = current_score
            best_colors = self.colors_oklch.copy()
        
        print(f"Initial: MinDist={current_min_dist:.4f}, StdDev={current_std_dev:.4f}, Score={current_score:.4f}")
        
        start_time = time.time()
        
        # Handle graceful shutdown
        def signal_handler(sig, frame):
            print("\nInterrupted! Saving state...")
            self.save_state(i, current_score, current_min_dist, current_std_dev, best_score, best_colors)
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            for i in range(self.start_iteration, self.iterations):
                temp = self.initial_temp * ((self.final_temp / self.initial_temp) ** (i / self.iterations))
                
                candidate_colors = self.colors_oklch.copy()
                idx_to_change = self.rng.integers(0, self.num_colors)
                
                perturbation = np.array([
                    self.rng.normal(0, 0.05),
                    self.rng.normal(0, 0.05),
                    self.rng.normal(0, 10.0)
                ])
                
                candidate_colors[idx_to_change] += perturbation
                
                # Normalize Hue and ensure positive Chroma
                candidate_colors[idx_to_change, 1] = np.abs(candidate_colors[idx_to_change, 1])
                candidate_colors[idx_to_change, 2] = candidate_colors[idx_to_change, 2] % 360.0
                
                candidate_colors[idx_to_change:idx_to_change+1] = self.constrain_to_gamut(candidate_colors[idx_to_change:idx_to_change+1])
                
                new_min_dist, new_std_dev = self.calculate_metrics(candidate_colors)
                new_score = self.calculate_score(new_min_dist, new_std_dev)
                
                # Maximize Score
                delta_score = new_score - current_score
                
                if delta_score > 0 or self.rng.random() < np.exp(delta_score / temp):
                    self.colors_oklch = candidate_colors
                    current_score = new_score
                    current_min_dist = new_min_dist
                    current_std_dev = new_std_dev
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_colors = self.colors_oklch.copy()
                
                if i % 10000 == 0:
                    elapsed_time = time.time() - start_time
                    if i > self.start_iteration:
                        iterations_per_sec = (i - self.start_iteration) / elapsed_time
                        remaining_iterations = self.iterations - i
                        eta_seconds = remaining_iterations / iterations_per_sec if iterations_per_sec > 0 else 0
                        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                    else:
                        eta_str = "Calculating..."
                    
                    progress = (i / self.iterations) * 100
                    print(f"Iter {i}/{self.iterations} ({progress:.1f}%): Temp={temp:.4f}, Score={current_score:.4f} (Min={current_min_dist:.4f}, Std={current_std_dev:.4f}) ETA: {eta_str}")

        except KeyboardInterrupt:
            print("\nKeyboardInterrupt caught. Saving state...")
            self.save_state(i, current_score, current_min_dist, current_std_dev, best_score, best_colors)
            sys.exit(0)
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            self.save_state(i, current_score, current_min_dist, current_std_dev, best_score, best_colors)
            raise e

        # Cleanup state file on success
        if os.path.exists(self.state_file):
            os.remove(self.state_file)
            print("Optimization finished. State file removed.")

        self.colors_oklch = best_colors
        print(f"Final Best Score: {best_score:.4f}")
        
        # Post-processing: TSP Sort
        self.sort_colors_tsp()

    def sort_colors_tsp(self):
        print("Sorting colors to minimize ring distance (TSP)...")
        oklab = self.oklch_to_oklab(self.colors_oklch)
        
        # Calculate full distance matrix
        diff = oklab[:, np.newaxis, :] - oklab[np.newaxis, :, :]
        dist_sq_matrix = np.sum(diff**2, axis=2)
        
        num_points = self.num_colors
        current_order = np.arange(num_points)
        
        def calculate_ring_dist(order):
            dist = 0
            for i in range(num_points):
                idx1 = order[i]
                idx2 = order[(i + 1) % num_points]
                dist += dist_sq_matrix[idx1, idx2]
            return dist
        
        current_dist = calculate_ring_dist(current_order)
        best_dist = current_dist
        best_order = current_order.copy()
        
        # Simple Simulated Annealing for TSP
        tsp_iterations = 10000
        tsp_temp = 1.0
        
        for i in range(tsp_iterations):
            temp = tsp_temp * (0.001 ** (i / tsp_iterations))
            
            # 2-opt move: reverse a segment
            new_order = current_order.copy()
            a = self.rng.integers(0, num_points)
            b = self.rng.integers(0, num_points)
            if a > b: a, b = b, a
            
            new_order[a:b+1] = new_order[a:b+1][::-1]
            
            new_dist = calculate_ring_dist(new_order)
            delta = current_dist - new_dist # We want to MINIMIZE distance, so positive delta is good
            
            if delta > 0 or self.rng.random() < np.exp(delta / temp):
                current_order = new_order
                current_dist = new_dist
                
                if current_dist < best_dist:
                    best_dist = current_dist
                    best_order = current_order.copy()
        
        print(f"TSP Sorted. Ring Distance Reduced: {best_dist:.4f}")
        self.colors_oklch = self.colors_oklch[best_order]

    def get_results(self):
        oklab = self.oklch_to_oklab(self.colors_oklch)
        linear = self.oklab_to_linear_srgb(oklab)
        srgb = self.linear_srgb_to_srgb(linear)
        srgb_8bit = np.clip(srgb * 255, 0, 255).astype(int)
        
        return self.colors_oklch, srgb_8bit

    def print_stats(self):
        min_dist, std_dev = self.calculate_metrics(self.colors_oklch)
        
        print("\n--- Statistics ---")
        print(f"Minimum Distance: {min_dist:.4f}")
        print(f"Std Dev of NN Distances: {std_dev:.4f}")

if __name__ == "__main__":
    opt = ColorOptimizer(num_colors=27)
    opt.optimize()
    
    oklch, srgb = opt.get_results()
    # opt.print_stats() # Removed in favor of manual formatting below
    
    # Calculate stats manually for output
    min_dist, std_dev = opt.calculate_metrics(oklch)
    
    output_lines = []
    output_lines.append("\n--- Statistics ---")
    output_lines.append(f"Minimum Distance: {min_dist:.4f}")
    output_lines.append(f"Std Dev of NN Distances: {std_dev:.4f}")
    
    output_lines.append("\n--- Final Colors ---")
    output_lines.append(f"{'Index':<6} {'Hex':<8} {'R,G,B':<12} {'L':<6} {'C':<6} {'H':<6}")
    for i in range(len(srgb)):
        r, g, b = srgb[i]
        l, c, h = oklch[i]
        hex_code = f"#{r:02x}{g:02x}{b:02x}"
        output_lines.append(f"{i:<6} {hex_code:<8} {r},{g},{b:<8} {l:.3f} {c:.3f} {h:.1f}")
    
    output_str = "\n".join(output_lines)
    print(output_str)
    
    with open("colors.txt", "w") as f:
        f.write(output_str)
    print("\nResults saved to colors.txt")
