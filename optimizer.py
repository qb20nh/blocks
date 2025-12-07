import numpy as np
import time
import json
import os
import signal
import sys
import argparse
import threading
import queue
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

@numba.jit(nopython=True)
def run_batch_optimization(
    colors_oklch, 
    colors_oklab,
    dist_sq,
    start_iter, 
    num_iters, 
    total_iters_for_temp, # For exponential schedule
    initial_temp, 
    final_temp, 
    phase, # 0=Normal, 1=Extended
    current_score, 
    current_min_dist, 
    current_std_dev,
    best_score, 
    best_colors, 
    best_min_dist,
    seed,
    ext_iter_offset=0,
    max_ext_iters=0,
    iters_since_improvement=0,
    last_improvement_val=0.0
):
    np.random.seed(seed)
    n = colors_oklch.shape[0]
    
    # Pre-allocate for temporary calculations
    old_row = np.empty(n, dtype=np.float64)
    
    for i in range(num_iters):
        current_iter = start_iter + i
        
        # Temperature Schedule
        if phase == 0: # Normal
            temp = initial_temp * ((final_temp / initial_temp) ** (current_iter / total_iters_for_temp))
        else: # Extended
            eff_ext_iter = current_iter # This is actually ext_iter
            progress = eff_ext_iter / max_ext_iters
            if progress >= 1.0:
                temp = 0.0
            else:
                temp = final_temp * (1.0 - progress)
                
        # Stop Condition for Extended
        if phase == 1:
            iters_since_improvement += 1
            if current_std_dev <= 0.0001 and iters_since_improvement > 10000:
                return (
                    current_score, current_min_dist, current_std_dev, 
                    best_score, best_min_dist, 
                    iters_since_improvement, last_improvement_val, 
                    True, i + 1 # Stop early, return iterations done
                )

        # Perturbation
        idx_to_change = np.random.randint(0, n)
        
        # Save old state
        old_color = colors_oklch[idx_to_change].copy()
        old_oklab = colors_oklab[idx_to_change].copy()
        # Save old distance row (only need one row as matrix is symmetric)
        for k in range(n):
            old_row[k] = dist_sq[idx_to_change, k]
            
        # Perturb
        perturbation = np.array([
            np.random.normal(0, 0.05),
            np.random.normal(0, 0.05),
            np.random.normal(0, 10.0)
        ])
        
        colors_oklch[idx_to_change] += perturbation
        
        # Normalize/Constrain
        colors_oklch[idx_to_change, 1] = np.abs(colors_oklch[idx_to_change, 1])
        colors_oklch[idx_to_change, 2] = colors_oklch[idx_to_change, 2] % 360.0
        
        # Constrain to gamut (in-place)
        # We need to call the function. Since it's JITed, it's fast.
        # But constrain_to_gamut takes array. We pass slice.
        # Numba slice passing is efficient.
        colors_oklch[idx_to_change:idx_to_change+1] = constrain_to_gamut(colors_oklch[idx_to_change:idx_to_change+1])
        
        # Update OKLAB for this color
        colors_oklab[idx_to_change] = oklch_to_oklab(colors_oklch[idx_to_change:idx_to_change+1])[0]
        
        # Incremental Distance Update
        # Update row/col idx_to_change in dist_sq
        for k in range(n):
            if k == idx_to_change:
                continue
            
            d = 0.0
            for c in range(3):
                diff = colors_oklab[idx_to_change, c] - colors_oklab[k, c]
                d += diff * diff
            
            dist_sq[idx_to_change, k] = d
            dist_sq[k, idx_to_change] = d
            
        # Recompute Metrics from dist_sq
        # We can optimize this too, but scanning N*N is fast enough for N=32
        min_dist = np.inf
        min_dists = np.empty(n, dtype=np.float64)
        
        for r in range(n):
            row_min = np.inf
            for c in range(n):
                if dist_sq[r, c] < row_min:
                    row_min = dist_sq[r, c]
            min_dists[r] = np.sqrt(row_min)
            if min_dists[r] < min_dist:
                min_dist = min_dists[r]
                
        std_dev = np.std(min_dists)
        new_score = min_dist - 1.0 * std_dev
        
        # Acceptance
        delta_score = new_score - current_score
        
        accept = False
        if delta_score > 0:
            accept = True
        elif temp > 0:
            if np.random.random() < np.exp(delta_score / temp):
                accept = True
                
        if accept:
            current_score = new_score
            current_min_dist = min_dist
            current_std_dev = std_dev
            
            if current_score > best_score:
                best_score = current_score
                best_colors[:] = colors_oklch[:] # Update best
                best_min_dist = current_min_dist
                
            if phase == 1:
                if current_min_dist > last_improvement_val + 0.0001:
                    last_improvement_val = current_min_dist
                    iters_since_improvement = 0
        else:
            # Revert
            colors_oklch[idx_to_change] = old_color
            colors_oklab[idx_to_change] = old_oklab
            for k in range(n):
                dist_sq[idx_to_change, k] = old_row[k]
                dist_sq[k, idx_to_change] = old_row[k]
                
    return (
        current_score, current_min_dist, current_std_dev, 
        best_score, best_min_dist, 
        iters_since_improvement, last_improvement_val, 
        False, num_iters
    )

class ColorOptimizer:
    def __init__(self, num_colors=27, iterations=None, tsp_iterations=None, force_fresh=False):
        self.num_colors = num_colors
        self.force_fresh = force_fresh
        self.rng = np.random.default_rng(42)
        
        # Auto-save queue and thread
        self.save_queue = queue.Queue()
        self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.save_thread.start()
        
        # Simulated Annealing parameters
        self.initial_temp = 1.0
        self.final_temp = 0.0001
        
        # Intelligent iteration scaling
        # Base: 27 colors -> 100M iterations
        base_colors = 27
        base_iterations = 100_000_000
        
        if iterations is not None:
            self.iterations = int(iterations)
        else:
            # Scale quadratically with number of colors
            scale_factor = (num_colors / base_colors) ** 2
            self.iterations = int(base_iterations * scale_factor)
            print(f"Auto-scaled iterations to {self.iterations:,} (Factor: {scale_factor:.2f}x)")

        # Ensure results directory exists
        os.makedirs("results", exist_ok=True)
        self.state_file = os.path.join("results", f"optimizer_state_{num_colors}.json")
        self.result_file = os.path.join("results", f"colors_{num_colors}.txt")
        self.start_iteration = 0
        
        # TSP parameters
        base_tsp_iterations = 100_000
        if tsp_iterations is not None:
            self.tsp_iterations = int(tsp_iterations)
        else:
             # Scale quadratically for TSP as well
            scale_factor = (num_colors / base_colors) ** 2
            self.tsp_iterations = int(base_tsp_iterations * scale_factor)
            print(f"Auto-scaled TSP iterations to {self.tsp_iterations:,}")
        
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

    def _save_worker(self):
        """Worker thread to handle save operations from the queue."""
        while True:
            try:
                args = self.save_queue.get()
                if args is None: # Sentinel
                    break
                self.save_state(*args, silent=True)
                self.save_queue.task_done()
            except Exception as e:
                # Just print error, don't crash thread
                print(f"Auto-save failed: {e}")

    def save_state(self, iteration, current_score, current_min_dist, current_std_dev, best_score, best_colors, best_min_dist, ext_iter=0, iters_since_improvement=0, last_improvement_val=0.0, silent=False):
        """Save the current state to a file."""
        if not silent:
            try:
                print(f"\nSaving state at iteration {iteration} (Ext: {ext_iter})...")
            except (BrokenPipeError, OSError):
                pass
            
        state = {
            'iteration': iteration,
            'colors_oklch': self.colors_oklch.tolist(),
            'current_score': float(current_score),
            'current_min_dist': float(current_min_dist),
            'current_std_dev': float(current_std_dev),
            'best_score': float(best_score),
            'best_colors': best_colors.tolist(),
            'best_min_dist': float(best_min_dist),
            'rng_state': self.rng.bit_generator.state,
            'ext_iter': int(ext_iter),
            'iters_since_improvement': int(iters_since_improvement),
            'last_improvement_val': float(last_improvement_val)
        }
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            # If we can't write to file, we are in trouble, but try to print error
            if not silent:
                try:
                    print(f"Error saving state: {e}")
                except:
                    pass
                
        if not silent:
            try:
                print("State saved.")
            except (BrokenPipeError, OSError):
                pass

    def load_state(self):
        """Load state from file if it exists."""
        if os.path.exists(self.state_file):
            print(f"Found saved state file: {self.state_file}")
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                
                self.start_iteration = state['iteration']
                self.colors_oklch = np.array(state['colors_oklch'])
                self.rng.bit_generator.state = state['rng_state']
                
                # Load extended phase state if available
                self.ext_iter_start = state.get('ext_iter', 0)
                self.iters_since_improvement_start = state.get('iters_since_improvement', 0)
                self.last_improvement_val_start = state.get('last_improvement_val', 0.0)
                
                # Restore best_colors if present (it should be)
                if 'best_colors' in state:
                    state['best_colors'] = np.array(state['best_colors'])
                
                print(f"Resuming from iteration {self.start_iteration} (Ext: {self.ext_iter_start})")
                return True, state
            except Exception as e:
                print(f"Failed to load state: {e}")
                return False, None
        return False, None

    def optimize(self):
        print(f"Optimizing {self.num_colors} colors using Hybrid Simulated Annealing...")
        
        # Try to load state
        loaded = False
        state = None
        if self.force_fresh:
            print("Force fresh start: Deleting existing state and results...")
            if os.path.exists(self.state_file):
                try:
                    os.remove(self.state_file)
                    print(f"Deleted state file: {self.state_file}")
                except OSError as e:
                    print(f"Error deleting state file: {e}")
            
            if os.path.exists(self.result_file):
                try:
                    os.remove(self.result_file)
                    print(f"Deleted result file: {self.result_file}")
                except OSError as e:
                    print(f"Error deleting result file: {e}")
        else:
            loaded, state = self.load_state()
        
        if loaded:
            current_score = state['current_score']
            current_min_dist = state['current_min_dist']
            current_std_dev = state['current_std_dev']
            best_score = state['best_score']
            best_colors = state['best_colors']
            best_min_dist = state.get('best_min_dist', current_min_dist) # Fallback for old states
        else:
            # Ensure valid start
            self.colors_oklch = self.constrain_to_gamut(self.colors_oklch)
            
            current_min_dist, current_std_dev = self.calculate_metrics(self.colors_oklch)
            current_score = self.calculate_score(current_min_dist, current_std_dev)
            
            best_score = current_score
            best_colors = self.colors_oklch.copy()
            best_min_dist = current_min_dist
        
        print(f"Initial: MinDist={current_min_dist:.4f}, StdDev={current_std_dev:.4f}, Score={current_score:.4f}")
        
        print(f"Initial: MinDist={current_min_dist:.4f}, StdDev={current_std_dev:.4f}, Score={current_score:.4f}")
        
        start_time = time.time()
        last_print_time = start_time
        
        # Initialize extended phase variables for signal handler scope
        ext_iter = getattr(self, 'ext_iter_start', 0)
        iters_since_improvement = getattr(self, 'iters_since_improvement_start', 0)
        last_improvement_val = getattr(self, 'last_improvement_val_start', best_min_dist)
        
        # Handle graceful shutdown
        def signal_handler(sig, frame):
            try:
                print("\nInterrupted! Saving state...")
            except (BrokenPipeError, OSError):
                pass
                
            try:
                current_iter = i
            except NameError:
                current_iter = self.start_iteration
            
            self.save_state(current_iter, current_score, current_min_dist, current_std_dev, best_score, best_colors, best_min_dist, ext_iter, iters_since_improvement, last_improvement_val)
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Windows Console Control Handler
        if sys.platform == 'win32':
            import ctypes
            from ctypes import wintypes
            
            def console_ctrl_handler(ctrl_type):
                # CTRL_C_EVENT = 0
                # CTRL_BREAK_EVENT = 1
                # CTRL_CLOSE_EVENT = 2
                
                # Let Python handle CTRL_C_EVENT (0) via KeyboardInterrupt/SIGINT
                if ctrl_type == 0:
                    return False
                
                if ctrl_type in (1, 2):
                    # Suppress print to avoid console lock contention during close
                    # try:
                    #     print(f"\nConsole event {ctrl_type} received. Saving state...")
                    # except (BrokenPipeError, OSError):
                    #     pass
                        
                    try:
                        try:
                            current_iter = i
                        except NameError:
                            current_iter = self.start_iteration
                            
                        self.save_state(current_iter, current_score, current_min_dist, current_std_dev, best_score, best_colors, best_min_dist, ext_iter, iters_since_improvement, last_improvement_val, silent=True)
                        
                        # For CTRL_CLOSE_EVENT, we must exit immediately as we are in a handler thread
                        # and the process is about to be killed.
                        import os
                        os._exit(0)
                        return True
                    except Exception as e:
                        # print(f"Error in console handler: {e}")
                        return False
                return False
            
            # Keep reference to avoid GC
            self._ctrl_handler = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_uint)(console_ctrl_handler)
            ctypes.windll.kernel32.SetConsoleCtrlHandler(self._ctrl_handler, True)

        # Initialize OKLAB and Distance Matrix for Numba
        self.colors_oklab = self.oklch_to_oklab(self.colors_oklch)
        
        # Calculate initial distance matrix
        n = self.num_colors
        self.dist_sq = np.empty((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                d = 0.0
                for k in range(3):
                    diff = self.colors_oklab[i, k] - self.colors_oklab[j, k]
                    d += diff * diff
                self.dist_sq[i, j] = d
            self.dist_sq[i, i] = np.inf

        try:
            # --- Main Optimization Phase ---
            chunk_size = 10000
            
            # Seed Numba RNG
            seed = self.rng.integers(0, 1000000)
            
            for i in range(self.start_iteration, self.iterations, chunk_size):
                iters_to_run = min(chunk_size, self.iterations - i)
                
                (current_score, current_min_dist, current_std_dev, 
                 best_score, best_min_dist, _, _, _, _) = run_batch_optimization(
                    self.colors_oklch, self.colors_oklab, self.dist_sq,
                    i, iters_to_run, self.iterations,
                    self.initial_temp, self.final_temp, 0, # Phase 0
                    current_score, current_min_dist, current_std_dev,
                    best_score, best_colors, best_min_dist,
                    seed + i # Vary seed slightly
                )
                
                # Auto-save check
                current_iter = i + iters_to_run
                if current_iter % 10_000_000 == 0:
                     # Create snapshot for thread safety
                     # We need to copy mutable arrays: best_colors
                     # Other values are immutable (int, float)
                     self.save_queue.put((
                         current_iter, current_score, current_min_dist, current_std_dev,
                         best_score, best_colors.copy(), best_min_dist,
                         0, 0, 0 # ext_iter, iters_since_imp, last_imp
                     ))
                
                # Update progress
                current_iter = i + iters_to_run
                current_time = time.time()
                if current_time - last_print_time > 2.0:
                    last_print_time = current_time
                    elapsed_time = current_time - start_time
                    temp = self.initial_temp * ((self.final_temp / self.initial_temp) ** (current_iter / self.iterations))
                    
                    if current_iter > self.start_iteration:
                        iterations_per_sec = (current_iter - self.start_iteration) / elapsed_time
                        remaining_iterations = self.iterations - current_iter
                        eta_seconds = remaining_iterations / iterations_per_sec if iterations_per_sec > 0 else 0
                        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                    else:
                        eta_str = "Calculating..."
                    
                    progress = (current_iter / self.iterations) * 100
                    print(f"Iter {current_iter}/{self.iterations} ({progress:.1f}%): Temp={temp:.4f}, Score={current_score:.4f} (Min={current_min_dist:.4f}, Std={current_std_dev:.4f}) Speed={iterations_per_sec:.2f} it/s ETA: {eta_str}")

            # --- Extended Optimization Phase ---
            print("\nEntering Extended Optimization Phase (Refining)...")
            print("Target: StdDev <= 0.0001 AND MinDist improvement < 0.0001 over 10k iters")
            
            max_ext_iters = self.iterations
            print(f"Extended Phase Limit: {max_ext_iters:,} iterations")
            
            if ext_iter == 0:
                last_improvement_val = best_min_dist
                
            while ext_iter < max_ext_iters:
                iters_to_run = min(chunk_size, max_ext_iters - ext_iter)
                
                (current_score, current_min_dist, current_std_dev, 
                 best_score, best_min_dist, 
                 iters_since_improvement, last_improvement_val, 
                 stop_early, iters_done) = run_batch_optimization(
                    self.colors_oklch, self.colors_oklab, self.dist_sq,
                    ext_iter + 1, iters_to_run, 0, # total_iters ignored in phase 1
                    0, self.final_temp, 1, # Phase 1
                    current_score, current_min_dist, current_std_dev,
                    best_score, best_colors, best_min_dist,
                    seed + ext_iter + 1000000,
                    ext_iter_offset=0,
                    max_ext_iters=max_ext_iters,
                    iters_since_improvement=iters_since_improvement,
                    last_improvement_val=last_improvement_val
                )
                
                ext_iter += iters_done
                
                # Auto-save check
                if ext_iter % 10_000_000 == 0:
                     self.save_queue.put((
                         self.iterations, current_score, current_min_dist, current_std_dev,
                         best_score, best_colors.copy(), best_min_dist,
                         ext_iter, iters_since_improvement, last_improvement_val
                     ))
                
                if stop_early:
                    print(f"\nExtended Phase Complete: Reached target StdDev ({current_std_dev:.6f}) AND MinDist plateaued.")
                    break
                    
                current_time = time.time()
                if current_time - last_print_time > 2.0:
                    last_print_time = current_time
                    # Calculate temp for display
                    progress = ext_iter / max_ext_iters
                    if progress >= 1.0:
                        disp_temp = 0.0
                    else:
                        disp_temp = self.final_temp * (1.0 - progress)
                    print(f"Ext Iter {ext_iter}: Temp={disp_temp:.4f}, Score={current_score:.4f} (Min={current_min_dist:.4f}, Std={current_std_dev:.4f})")

        except KeyboardInterrupt:
            print("\nKeyboardInterrupt caught. Saving state...")
            current_iter = i if 'i' in locals() else self.start_iteration
            self.save_state(current_iter, current_score, current_min_dist, current_std_dev, best_score, best_colors, best_min_dist, ext_iter, iters_since_improvement, last_improvement_val)
            sys.exit(0)
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            current_iter = i if 'i' in locals() else self.start_iteration
            self.save_state(current_iter, current_score, current_min_dist, current_std_dev, best_score, best_colors, best_min_dist, ext_iter, iters_since_improvement, last_improvement_val)
            raise e

        # Cleanup state file on success
        if os.path.exists(self.state_file):
            os.remove(self.state_file)
            print("Optimization finished. State file removed.")

        self.colors_oklch = best_colors
        self.ext_iterations = ext_iter
        self.total_iterations = self.iterations + ext_iter
        print(f"Final Best Score: {best_score:.4f}")
        print(f"Final Best Score: {best_score:.4f}")
        
        # Post-processing: TSP Sort
        self.sort_colors_tsp()
        
        # Post-processing: Canonicalize Order
        self.canonicalize_order()

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
        tsp_iterations = self.tsp_iterations
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

    def canonicalize_order(self):
        print("Canonicalizing color order...")
        # 1. Base: black (or lowest L color) is index 0
        l_values = self.colors_oklch[:, 0]
        min_l_idx = np.argmin(l_values)
        
        # Rotate so min_l_idx is at 0
        self.colors_oklch = np.roll(self.colors_oklch, -min_l_idx, axis=0)
        
        # 2. Direction: after black, color change should be ascending order in hue.
        # Calculate H_w = H*sqrt(L*C)
        h = self.colors_oklch[:, 2]
        l = self.colors_oklch[:, 0]
        c = self.colors_oklch[:, 1]
        
        # Ensure H is positive for calculation (it should be from 0-360)
        h = h % 360.0
        
        # Use distance from 0.5 for lightness weight
        l_weight = 0.5 - np.abs(l - 0.5)
        h_w = h * np.sqrt(l_weight * c)
        
        # We consider the sequence after the first element (index 0)
        # Sequence indices: 1 to N-1
        n = self.num_colors
        if n <= 2:
            return # Nothing to order direction-wise
            
        # Split into two halves
        # If N=27, remaining is 26. Half is 13.
        # First half: 1 to 13. Second half: 14 to 26.
        num_remaining = n - 1
        half_size = num_remaining // 2
        
        first_half_sum = np.sum(h_w[1 : 1 + half_size])
        second_half_sum = np.sum(h_w[1 + half_size :])
        
        print(f"Direction Check: First Half Sum = {first_half_sum:.4f}, Second Half Sum = {second_half_sum:.4f}")
        
        # Choose the direction where the sum H_w of first half is less than second half
        if first_half_sum > second_half_sum:
            print("Reversing direction to satisfy canonical order...")
            # Reverse elements from 1 to end
            self.colors_oklch[1:] = self.colors_oklch[1:][::-1]
            
            # Re-verify (optional, for debug)
            # h_new = self.colors_oklch[:, 2] % 360.0
            # l_new = self.colors_oklch[:, 0]
            # c_new = self.colors_oklch[:, 1]
            # h_w_new = h_new * np.sqrt(l_new * c_new)
            # s1 = np.sum(h_w_new[1 : 1 + half_size])
            # s2 = np.sum(h_w_new[1 + half_size :])
            # print(f"New sums: {s1:.4f} vs {s2:.4f}")
        else:
            print("Direction is already canonical.")

    def get_results(self):
        oklab = self.oklch_to_oklab(self.colors_oklch)
        linear = self.oklab_to_linear_srgb(oklab)
        srgb = self.linear_srgb_to_srgb(linear)
        srgb_8bit = np.clip(np.round(srgb * 255), 0, 255).astype(int)
        
        return self.colors_oklch, srgb_8bit

    def print_stats(self):
        min_dist, std_dev = self.calculate_metrics(self.colors_oklch)
        
        print("\n--- Statistics ---")
        print(f"Minimum Distance: {min_dist:.5f}")
        print(f"Std Dev of NN Distances: {std_dev:.5f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize colors in OKLCH space.")
    parser.add_argument("-n", "--num_colors", type=int, required=True, help="Number of colors to optimize")
    parser.add_argument("--iterations", type=int, help="Number of SA iterations (default: auto-scaled)")
    parser.add_argument("--tsp_iterations", type=int, help="Number of TSP iterations (default: auto-scaled)")
    parser.add_argument("-f", "--force", action="store_true", help="Force start fresh (ignore existing state)")
    
    args = parser.parse_args()
    
    try:
        opt = ColorOptimizer(num_colors=args.num_colors, iterations=args.iterations, tsp_iterations=args.tsp_iterations, force_fresh=args.force)
        opt.optimize()
        
        oklch, srgb = opt.get_results()
        # opt.print_stats() # Removed in favor of manual formatting below
        
        # Calculate stats manually for output
        min_dist, std_dev = opt.calculate_metrics(oklch)
        
        output_lines = []
        output_lines.append("--- Iterations ---")
        output_lines.append(f"Total Iterations: {opt.total_iterations:,}")
        output_lines.append(f"Ext Iterations: {opt.ext_iterations:,}")
        output_lines.append(f"TSP Iterations: {opt.tsp_iterations:,}")

        output_lines.append("\n--- Statistics ---")
        output_lines.append(f"Minimum Distance: {min_dist:.5f}")
        output_lines.append(f"Std Dev of NN Distances: {std_dev:.5f}")
        
        output_lines.append("\n--- Final Colors ---")
        output_lines.append(f"{'Index':<6} {'Hex':<8} {'R,G,B':<14} {'L':<8} {'C':<8} {'H':<8}".rstrip())
        for i in range(len(srgb)):
            r, g, b = srgb[i]
            l, c, h = oklch[i]
            hex_code = f"#{r:02x}{g:02x}{b:02x}"
            rgb_str = f"{r},{g},{b}"
            output_lines.append(f"{i:<6} {hex_code:<8} {rgb_str:<14} {l:<8.3f} {c:<8.3f} {h:<8.1f}".rstrip())
        
        output_str = "\n".join(output_lines)
        print(output_str)
        
        filename = os.path.join("results", f"colors_{args.num_colors}.txt")
        with open(filename, "w") as f:
            f.write(output_str)
        print(f"\nResults saved to {filename}")
    except KeyboardInterrupt:
        print("\nOptimization cancelled by user.")
        sys.exit(0)
