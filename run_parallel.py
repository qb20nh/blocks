import multiprocessing
import subprocess
import os
import sys
import time
import threading
import shutil

# ANSI escape codes
ANSI_RESET = "\033[0m"
ANSI_CLEAR_SCREEN = "\033[2J"
ANSI_HOME = "\033[H"
ANSI_CURSOR_UP = "\033[A"
ANSI_CLEAR_LINE = "\033[K"

def init_worker(q):
    """Initialize worker with the queue."""
    global queue
    queue = q

def run_optimizer(n):
    """Runs the optimizer for a specific number of colors."""
    # Set environment variables to ensure single-core execution per task
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    
    # Check if already done
    # Check if already done
    result_file = os.path.join("results", f"colors_{n}.txt")
    state_file = os.path.join("results", f"optimizer_state_{n}.pkl")
    if os.path.exists(result_file) and not os.path.exists(state_file):
        queue.put((n, "Skipped (Already done)"))
        return n, True, "Skipped"

    start_time = time.time()
    try:
        queue.put((n, "Starting..."))
        
        # Set process priority to HIGH_PRIORITY_CLASS (0x00000080) on Windows
        creationflags = 0
        if sys.platform == "win32":
            creationflags = getattr(subprocess, "HIGH_PRIORITY_CLASS", 0x00000080)

        # Run the optimizer script with unbuffered output
        process = subprocess.Popen(
            [sys.executable, "-u", "optimizer.py", "-n", str(n)],
            env=env,
            creationflags=creationflags,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        for line in process.stdout:
            line = line.strip()
            if line:
                if "Ext Iter" in line:
                    try:
                        # Example: Ext Iter 1: Temp=..., Score=... (Min=..., Std=...)
                        # Shorten to: Ext 1: Score=...
                        parts = line.split(":")
                        if len(parts) > 1:
                             # Try to extract score
                             score_part = line.split("Score=")[1].split("(")[0].strip()
                             queue.put((n, f"Ext {parts[0].split(' ')[2]}: Score={score_part}"))
                        else:
                            queue.put((n, "Ext Refining..."))
                    except:
                        queue.put((n, line))
                elif "Iter" in line:
                    try:
                        # Extract progress info to keep it short
                        # Example: Iter 100/1000 (10.0%): Temp=..., Score=... ETA: ...
                        parts = line.split("):")
                        if len(parts) > 1:
                            queue.put((n, parts[0] + ") " + parts[1].split("ETA:")[1].strip() if "ETA:" in parts[1] else line))
                        else:
                            queue.put((n, line))
                    except:
                        queue.put((n, line))
                elif "Extended Phase Complete" in line:
                    queue.put((n, "Ext Phase Done"))
                elif "Final" in line:
                    queue.put((n, "Finalizing..."))
                elif "TSP" in line:
                    queue.put((n, "TSP Sorting..."))
        
        process.wait()
        
        if process.returncode != 0:
            # Capture remaining output
            remaining_output = process.stdout.read()
            full_error = f"Failed! Return Code: {process.returncode}\nOutput:\n{remaining_output}"
            queue.put((n, "Failed!"))
            # Print to stderr so we can see it
            sys.stderr.write(f"\nTask n={n} failed:\n{full_error}\n")
            return n, False, full_error
            
        duration = time.time() - start_time
        queue.put((n, f"Done in {duration:.2f}s"))
        return n, True, ""
        
    except Exception as e:
        queue.put((n, f"Error: {str(e)}"))
        return n, False, str(e)

def display_manager(queue, num_tasks, stop_event):
    """Manages the display of progress lines."""
    # Clear screen initially
    sys.stdout.write(ANSI_CLEAR_SCREEN)
    sys.stdout.write(ANSI_HOME)
    sys.stdout.flush()
    
    status_map = {n: "Waiting..." for n in range(3, 33)}
    
    # Determine layout
    columns, lines = shutil.get_terminal_size()
    
    # Calculate available width for status messages
    # Format: "n=XX: {msg} | n=XX: {msg}"
    # Fixed chars: "n=XX: " (6) * 2 + " | " (3) = 15 chars
    # We want to leave a little buffer, say 5 chars
    available_width = columns - 20
    col_width = max(35, available_width // 2)

    while not stop_event.is_set() or not queue.empty():
        try:
            # Process all available messages
            while True:
                n, msg = queue.get_nowait()
                status_map[n] = msg
        except:
            pass
            
        # Redraw
        # Move to home
        sys.stdout.write(ANSI_HOME)
        print(f"Parallel Optimizer Progress ({multiprocessing.cpu_count()} cores)")
        print("-" * columns)
        
        # Print tasks in two columns if possible
        half = (33 - 3 + 1) // 2 + (33 - 3 + 1) % 2
        for i in range(half):
            n1 = 3 + i
            n2 = 3 + i + half
            
            s1 = f"n={n1:<2}: {status_map.get(n1, '')[:col_width]:<{col_width}}"
            s2 = ""
            if n2 <= 32:
                s2 = f" | n={n2:<2}: {status_map.get(n2, '')[:col_width]:<{col_width}}"
            
            print(f"{s1}{s2}" + ANSI_CLEAR_LINE)
            
        sys.stdout.flush()
        time.sleep(0.5)

def warmup():
    """Runs a quick optimization to ensure Numba compilation is cached."""
    print("Warming up (compiling Numba functions)...")
    try:
        # Ensure warmup doesn't get skipped
        if os.path.exists(os.path.join("results", "colors_2.txt")):
            os.remove(os.path.join("results", "colors_2.txt"))
        # Run a very short optimization
        subprocess.run(
            [sys.executable, "optimizer.py", "-n", "2", "--iterations", "1", "--tsp_iterations", "1"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print("Warmup complete.")
        # Remove the artifact file so n=3 doesn't get skipped
        if os.path.exists(os.path.join("results", "colors_2.txt")):
            os.remove(os.path.join("results", "colors_2.txt"))

    except subprocess.CalledProcessError:
        print("Warmup failed! Proceeding anyway, but errors may occur.")

import ctypes

def prevent_sleep():
    """Context manager to prevent system sleep on Windows."""
    class PreventSleep:
        def __enter__(self):
            if sys.platform == 'win32':
                # ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
                # 0x80000000 | 0x00000001 | 0x00000002
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if sys.platform == 'win32':
                # ES_CONTINUOUS
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    
    return PreventSleep()

def main():
    # Warmup to prevent race conditions in Numba compilation
    warmup()

    # Range from 3 to 32 (inclusive)
    nums = list(range(3, 33))
    num_cores = os.cpu_count()
    
    # Use standard multiprocessing.Queue instead of Manager
    queue = multiprocessing.Queue()
    stop_event = threading.Event()
    
    # Start display thread
    display_thread = threading.Thread(target=display_manager, args=(queue, len(nums), stop_event))
    display_thread.start()
    
    try:
        with prevent_sleep():
            # Use initializer to pass queue to workers
            with multiprocessing.Pool(processes=num_cores, initializer=init_worker, initargs=(queue,)) as pool:
                # Map just the numbers, worker uses global queue
                results = pool.map(run_optimizer, nums)
    finally:
        stop_event.set()
        display_thread.join()
            
    print("\nAll tasks completed.")

if __name__ == "__main__":
    # Enable ANSI support on Windows if needed (Python 3.8+ does this automatically mostly)
    os.system("") 
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)
