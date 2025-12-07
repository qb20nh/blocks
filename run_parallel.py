import multiprocessing
import subprocess
import os
import sys
import time
import threading
import shutil
import argparse

# Priority Map for Windows
PRIORITY_MAP = {
    'low': 0x00000040, # IDLE_PRIORITY_CLASS
    'below_normal': 0x00004000, # BELOW_NORMAL_PRIORITY_CLASS
    'normal': 0x00000020, # NORMAL_PRIORITY_CLASS
    'above_normal': 0x00008000, # ABOVE_NORMAL_PRIORITY_CLASS
    'high': 0x00000080, # HIGH_PRIORITY_CLASS
    'realtime': 0x00000100 # REALTIME_PRIORITY_CLASS
}

def parse_range(range_str):
    """Parses a range string (e.g., '3-5,10') into a sorted list of integers."""
    nums = set()
    parts = range_str.split(',')
    for part in parts:
        part = part.strip()
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                nums.update(range(start, end + 1))
            except ValueError:
                print(f"Error: Invalid range format '{part}'")
                sys.exit(1)
        else:
            try:
                nums.add(int(part))
            except ValueError:
                print(f"Error: Invalid number '{part}'")
                sys.exit(1)
    return sorted(list(nums))

# ANSI escape codes
ANSI_RESET = "\033[0m"
ANSI_CLEAR_SCREEN = "\033[2J"
ANSI_HOME = "\033[H"
ANSI_CURSOR_UP = "\033[A"
ANSI_CLEAR_LINE = "\033[K"

def init_worker(q, priority_level, force_fresh, abort_val):
    """Initialize worker with the queue and settings."""
    global queue, WORKER_PRIORITY, FORCE_FRESH, ABORT_VAL
    queue = q
    WORKER_PRIORITY = priority_level
    FORCE_FRESH = force_fresh
    ABORT_VAL = abort_val

def run_optimizer(n):
    """Runs the optimizer for a specific number of colors."""
    # Check abort value (safe to read shared memory)
    if ABORT_VAL.value == 1:
        return n, False, "Aborted"
        
    # Set environment variables to ensure single-core execution per task
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    
    # Check if already done
    result_file = os.path.join("results", f"colors_{n}.txt")
    state_file = os.path.join("results", f"optimizer_state_{n}.pkl")
    
    if FORCE_FRESH:
        pass # Skip check, let optimizer handle it
    elif os.path.exists(result_file) and not os.path.exists(state_file):
        queue.put((n, "Skipped (Already done)"))
        return n, True, "Skipped"

    start_time = time.time()
    try:
        queue.put((n, "Starting..."))
        
        # Set process priority
        creationflags = 0
        if sys.platform == "win32":
            creationflags = WORKER_PRIORITY

        # Build command
        cmd = [sys.executable, "-u", "optimizer.py", "-n", str(n)]
        if FORCE_FRESH:
            cmd.append("--force")

        # Run the optimizer script with unbuffered output
        try:
            process = subprocess.Popen(
                cmd,
                env=env,
                creationflags=creationflags,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
        except (OSError, ValueError) as e:
            # Catch errors like "Duplicate handle" or process creation failures during shutdown
            return n, False, f"Spawn Failed: {e}"
        
        # Non-blocking output reader
        import queue as std_queue # Rename to avoid conflict with global 'queue'
        line_queue = std_queue.Queue()
        
        def reader_thread(proc, q):
            try:
                for line in proc.stdout:
                    q.put(line)
            except:
                pass
            finally:
                q.put(None) # Sentinel

        t = threading.Thread(target=reader_thread, args=(process, line_queue))
        t.daemon = True # Ensure thread dies if main process dies
        t.start()

        try:
            while True:
                # Check abort signal
                if ABORT_VAL.value == 1:
                    process.terminate()
                    return n, False, "Aborted"

                # Check if process is done
                ret = process.poll()
                
                # Try to read lines (non-blocking w/ timeout)
                try:
                    # Wait a bit for output, but yield frequently to check abort
                    line = line_queue.get(timeout=0.1)
                except std_queue.Empty:
                    # No output yet
                    if ret is not None:
                        # Process finished and queue empty
                        break
                    continue

                if line is None: # Sentinel
                    break

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
                                # Parse speed if present
                                if "Speed=" in line:
                                    try:
                                        speed_str = line.split("Speed=")[1].split("it/s")[0].strip()
                                        speed = float(speed_str)
                                        queue.put(('SPEED', n, speed))
                                    except:
                                        pass
                                
                                # Clean up line for display
                                display_line = parts[0] + ") " + parts[1].split("ETA:")[1].strip() if "ETA:" in parts[1] else line
                                # Remove Speed=... from display line if present to keep it short
                                if "Speed=" in display_line:
                                    pre_speed = display_line.split("Speed=")[0].strip()
                                    post_speed = display_line.split("it/s")[1].strip() if "it/s" in display_line else ""
                                    display_line = f"{pre_speed} {post_speed}".strip()
                                    
                                queue.put((n, display_line))
                            else:
                                queue.put((n, line))
                        except:
                            queue.put((n, line))
                    elif "Extended Phase Complete" in line:
                        queue.put((n, "Ext Phase Done"))
                    elif "Final" in line:
                        queue.put((n, "Finalizing..."))
                    elif "Sorting colors" in line:
                        queue.put((n, "TSP Sorting..."))
                    elif "Force fresh start" in line:
                        queue.put((n, "Cleaning..."))
                    elif "Resuming" in line:
                        queue.put((n, "Resuming..."))
                    elif "Optimizing" in line and "colors" in line:
                        queue.put((n, "Initializing..."))
            
            process.wait()
            
        except KeyboardInterrupt:
            # On interrupt, we want to wait for the child to finish saving
            queue.put((n, "Stopping (Saving)..."))
            # The child process should also receive the signal and start saving.
            # We just wait for it to exit.
            process.wait()
            return n, False, "Interrupted"
        
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

def display_manager(queue, nums, num_threads, stop_event):
    """Manages the display of progress lines."""
    # Clear screen initially
    sys.stdout.write(ANSI_CLEAR_SCREEN)
    sys.stdout.write(ANSI_HOME)
    sys.stdout.flush()
    
    status_map = {n: "Waiting..." for n in nums}
    speed_map = {n: 0.0 for n in nums}
    
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
                item = queue.get_nowait()
                if isinstance(item, tuple) and len(item) == 3 and item[0] == 'SPEED':
                    _, n, speed = item
                    speed_map[n] = speed
                else:
                    n, msg = item
                    status_map[n] = msg
        except:
            pass
            
        # Redraw
        # Move to home
        sys.stdout.write(ANSI_HOME)
        total_speed = sum(speed_map.values())
        title = f"Parallel Optimizer Progress ({num_threads} threads)"
        speed_info = f"Total: {total_speed:,.0f} it/s"
        padding = columns - len(title) - len(speed_info) - 2
        if padding > 0:
            print(f"{title}{' ' * padding}{speed_info}")
        else:
            print(title)
        print("-" * columns)
        
        # Print tasks in two columns if possible
        num_tasks = len(nums)
        half = (num_tasks + 1) // 2
        
        for i in range(half):
            n1 = nums[i]
            s1 = f"n={n1:<2}: {status_map.get(n1, '')[:col_width]:<{col_width}}"
            
            s2 = ""
            if i + half < num_tasks:
                n2 = nums[i + half]
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
    parser = argparse.ArgumentParser(description="Run parallel color optimization.")
    parser.add_argument("-t", "--threads", type=int, default=os.cpu_count(), help="Number of threads (processes) to use.")
    parser.add_argument("-p", "--priority", type=str, choices=PRIORITY_MAP.keys(), default="high", help="Process priority.")
    parser.add_argument("-r", "--range", type=str, required=True, help="Range of N values (e.g., '3-32', '5,10-15').")
    parser.add_argument("-f", "--force", action="store_true", help="Force start fresh (ignore/delete existing results).")
    
    args = parser.parse_args()
    
    # Validation
    cpu_count = os.cpu_count()
    if args.threads > cpu_count:
        print(f"Error: Requested threads ({args.threads}) exceeds CPU count ({cpu_count}).")
        sys.exit(1)
        
    if args.priority == 'realtime' and args.threads >= cpu_count:
        print(f"Error: Cannot run with realtime priority and threads >= CPU count ({cpu_count}). This will freeze your system.")
        sys.exit(1)
        
    # Confirmation for High Priority + All Cores
    if args.priority == 'high' and args.threads == cpu_count:
        print(f"Warning: Running with HIGH priority on ALL cores ({cpu_count}). System responsiveness will drop dramatically.")
        response = input("Continue? (y/n): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Operation cancelled.")
            sys.exit(0)

    # Parse range
    nums = parse_range(args.range)
    if not nums:
        print("Error: No valid numbers in range.")
        sys.exit(1)

    # Cap threads to number of tasks
    num_tasks = len(nums)
    if args.threads > num_tasks:
        print(f"Info: Requested threads ({args.threads}) > tasks ({num_tasks}). Reducing threads to {num_tasks}.")
        args.threads = num_tasks

    # Warmup to prevent race conditions in Numba compilation
    warmup()

    # Use standard multiprocessing.Queue instead of Manager
    queue = multiprocessing.Queue()
    stop_event = threading.Event()
    # Use multiprocessing.Value for robust abort signaling
    abort_val = multiprocessing.Value('i', 0)
    
    # Start display thread
    display_thread = threading.Thread(target=display_manager, args=(queue, nums, args.threads, stop_event))
    display_thread.start()
    
    priority_val = PRIORITY_MAP[args.priority]
    
    # Windows Console Control Handler for Main Process
    # We need to define it here to access 'pool' variable which we will initialize shortly
    pool = None
    
    if sys.platform == 'win32':
        import ctypes
        def console_ctrl_handler(ctrl_type):
            if ctrl_type == 2: # CTRL_CLOSE_EVENT
                print("\nMain process received CTRL_CLOSE_EVENT. Waiting for workers...")
                # Signal workers to abort new tasks
                abort_val.value = 1
                
                # If pool is running, close and join it to wait exactly as long as needed
                if pool:
                    try:
                        pool.close()
                        pool.join()
                    except:
                        pass
                
                # Force exit immediately after workers are done
                os._exit(0)
                return True
            return False
            
        _ctrl_handler = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_uint)(console_ctrl_handler)
        ctypes.windll.kernel32.SetConsoleCtrlHandler(_ctrl_handler, True)

    try:
        with prevent_sleep():
            # Use initializer to pass queue to workers
            # We assign to the outer 'pool' variable so the handler can access it
            pool = multiprocessing.Pool(processes=args.threads, initializer=init_worker, initargs=(queue, priority_val, args.force, abort_val))
            try:
                # Map just the numbers, worker uses global queue
                results = pool.map(run_optimizer, nums)
                pool.close()
                pool.join()
            except KeyboardInterrupt:
                print("\nMain process interrupted. Waiting for workers to save state...")
                # Signal workers to abort new tasks
                abort_val.value = 1
                # Explicitly close and join the pool to allow workers to finish their cleanup.
                # This prevents the context manager from calling terminate() immediately.
                pool.close()
                pool.join()
            except Exception:
                pool.terminate()
                raise
            finally:
                # Ensure pool is closed if not already
                try:
                    pool.close()
                    pool.join()
                except:
                    pass
    finally:
        stop_event.set()
        display_thread.join()
            
    if abort_val.value == 1:
        print("\nOptimization interrupted.")
    else:
        print("\nAll tasks completed.")

if __name__ == "__main__":
    # Enable ANSI support on Windows if needed (Python 3.8+ does this automatically mostly)
    os.system("") 
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)
