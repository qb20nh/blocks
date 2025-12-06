import os
import sys
import json
import pickle
import multiprocessing
import ctypes
import glob
import time
import numpy as np
from ctypes import wintypes

# --- Windows Job Object Constants & Structures ---
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
JOB_OBJECT_UILIMIT_ALL = 0x000000FF

class IO_COUNTERS(ctypes.Structure):
    _fields_ = [('ReadOperationCount', ctypes.c_ulonglong),
                ('WriteOperationCount', ctypes.c_ulonglong),
                ('OtherOperationCount', ctypes.c_ulonglong),
                ('ReadTransferCount', ctypes.c_ulonglong),
                ('WriteTransferCount', ctypes.c_ulonglong),
                ('OtherTransferCount', ctypes.c_ulonglong)]

class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [('PerProcessUserTimeLimit', ctypes.c_longlong),
                ('PerJobUserTimeLimit', ctypes.c_longlong),
                ('LimitFlags', ctypes.c_ulong),
                ('MinimumWorkingSetSize', ctypes.c_size_t),
                ('MaximumWorkingSetSize', ctypes.c_size_t),
                ('ActiveProcessLimit', ctypes.c_ulong),
                ('Affinity', ctypes.c_size_t),
                ('PriorityClass', ctypes.c_ulong),
                ('SchedulingClass', ctypes.c_ulong)]

class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [('BasicLimitInformation', JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ('IoInfo', IO_COUNTERS),
                ('ProcessMemoryLimit', ctypes.c_size_t),
                ('JobMemoryLimit', ctypes.c_size_t),
                ('PeakProcessMemoryUsed', ctypes.c_size_t),
                ('PeakJobMemoryUsed', ctypes.c_size_t)]

class JOBOBJECT_BASIC_UI_RESTRICTIONS(ctypes.Structure):
    _fields_ = [('UIRestrictionsClass', ctypes.c_ulong)]

# --- Safe Unpickler ---
class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Strict Whitelist
        allowed = {
            ('builtins', 'dict'), ('builtins', 'list'), ('builtins', 'int'), 
            ('builtins', 'float'), ('builtins', 'str'), ('builtins', 'tuple'),
            ('builtins', 'set'), ('builtins', 'frozenset'), ('builtins', 'bool'),
            ('builtins', 'NoneType'),
            ('numpy', 'dtype'),
            ('numpy.core.multiarray', '_reconstruct'),
            ('numpy._core.numeric', '_frombuffer'), # NumPy 2.x
            ('numpy.random._generator', 'Generator'),
            ('numpy.random._pcg64', 'PCG64'),
            # NumPy < 2.0 compatibility if needed
            ('numpy.core.numeric', '_frombuffer'),
        }
        
        if (module, name) in allowed:
            return super().find_class(module, name)
        
        raise pickle.UnpicklingError(f"Forbidden global: {module}.{name}")

# --- Audit Hook ---
def audit_hook(event, args):
    # Block dangerous events
    if event in ('open', 'socket.socket', 'subprocess.Popen', 'os.system', 'os.spawn', 'os.posix_spawn'):
        raise RuntimeError(f"Action blocked by sandbox: {event}")
    
    # Block ctypes events
    if event.startswith('ctypes.'):
        raise RuntimeError(f"Action blocked by sandbox: {event}")
    
    # Intercept pickle.find_class for defense-in-depth
    if event == 'pickle.find_class':
        module, name = args
        allowed = {
            ('builtins', 'dict'), ('builtins', 'list'), ('builtins', 'int'), 
            ('builtins', 'float'), ('builtins', 'str'), ('builtins', 'tuple'),
            ('builtins', 'set'), ('builtins', 'frozenset'), ('builtins', 'bool'),
            ('builtins', 'NoneType'),
            ('numpy', 'dtype'),
            ('numpy.core.multiarray', '_reconstruct'),
            ('numpy._core.numeric', '_frombuffer'),
            ('numpy.random._generator', 'Generator'),
            ('numpy.random._pcg64', 'PCG64'),
            ('numpy.core.numeric', '_frombuffer'),
        }
        if (module, name) not in allowed:
             raise RuntimeError(f"Pickle find_class blocked by sandbox: {module}.{name}")
             
    # Monitor attribute access (optional, can be noisy, but good for dunder checks if args provided)
    # object.__getattr__ is not a standard audit event, but some are.
    
    # Block import of new modules
    if event == 'import':
        # args is (module, filename, sys.path, site)
        # We can't easily whitelist all imports needed for startup, 
        # but we can block specific dangerous ones if they try to happen LATE.
        # However, since we scrub sys.modules, any import attempt is suspicious.
        # Let's block 'os', 'subprocess', 'socket' specifically.
        module_name = args[0]
        if module_name in ('os', 'subprocess', 'socket', 'shutil', 'http', 'ctypes'):
             raise RuntimeError(f"Import blocked by sandbox: {module_name}")

# --- Child Process Logic ---


def safe_load_process(file_content, pipe_conn):
    try:
        # Apply Sandbox (Job Object + Audit Hook + Scrubbing)
        
        # Job Object
        kernel32 = ctypes.windll.kernel32
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            raise RuntimeError(f"Failed to create Job Object. Error: {ctypes.GetLastError()}")

        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_JOB_MEMORY | JOB_OBJECT_LIMIT_ACTIVE_PROCESS | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        info.JobMemoryLimit = 100 * 1024 * 1024 
        info.BasicLimitInformation.ActiveProcessLimit = 1
        ui_info = JOBOBJECT_BASIC_UI_RESTRICTIONS()
        ui_info.UIRestrictionsClass = JOB_OBJECT_UILIMIT_ALL
        
        if not ctypes.windll.kernel32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
             raise RuntimeError(f"Failed to set Job Object limits. Error: {ctypes.GetLastError()}")
             
        if not ctypes.windll.kernel32.SetInformationJobObject(job, 4, ctypes.byref(ui_info), ctypes.sizeof(ui_info)):
             raise RuntimeError(f"Failed to set Job Object UI limits. Error: {ctypes.GetLastError()}")
             
        if not ctypes.windll.kernel32.AssignProcessToJobObject(job, ctypes.windll.kernel32.GetCurrentProcess()):
             raise RuntimeError(f"Failed to assign process to Job Object. Error: {ctypes.GetLastError()}")

        # Scrubbing - Remove from sys.modules AND global scope
        sensitive_modules = ['os', 'subprocess', 'socket', 'shutil', 'ctypes', 'multiprocessing']
        for mod in sensitive_modules:
            if mod in sys.modules:
                del sys.modules[mod]
            # Also remove from global scope if present
            if mod in globals():
                del globals()[mod]
                
        safe_builtins = sys.modules['builtins']
        for name in ['open', 'exec', 'eval']:
            if hasattr(safe_builtins, name):
                delattr(safe_builtins, name)

        # Secure __import__ wrapper
        original_import = __import__
        def secure_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == 'numpy' or name.startswith('numpy.'):
                return original_import(name, globals, locals, fromlist, level)
            safe_modules = {'math', 'struct', 'binascii', 'codecs', 'encodings', 'collections', 'itertools', 'functools', 'copyreg', 're', 'contextlib', 'enum', 'warnings'}
            if name in safe_modules or any(name.startswith(m + '.') for m in safe_modules):
                 return original_import(name, globals, locals, fromlist, level)
            raise ImportError(f"Import of '{name}' is forbidden by sandbox.")
        safe_builtins.__import__ = secure_import

        # Audit Hook
        sys.addaudithook(audit_hook)

        # Unpickle
        data = SafeUnpickler(io.BytesIO(file_content)).load()
        
        # Convert to JSON-serializable
        # We need to handle numpy arrays
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            if isinstance(obj, tuple):
                return tuple(convert(v) for v in obj)
            # Handle RNG state (BitGenerator)
            if hasattr(obj, 'state') and isinstance(obj.state, dict):
                 return {'rng_state': obj.state} # Simplified for now, or just extract state dict
            return obj

        # Special handling for our specific state structure
        # rng_state in numpy is a dict (for PCG64) or tuple (MT19937)
        
        serializable_data = {}
        for k, v in data.items():
            if k == 'rng_state':
                # It's a dict from bit_generator.state
                serializable_data[k] = v # It is already a dict of ints/dicts
            else:
                serializable_data[k] = convert(v)
        
        # Secure IPC: Send bytes (JSON) instead of pickling
        response = json.dumps({'success': True, 'data': serializable_data}).encode('utf-8')
        pipe_conn.send_bytes(response)
        
    except Exception as e:
        # We might not be able to send if pipe is broken or blocked, but try
        try:
            error_response = json.dumps({'success': False, 'error': str(e)}).encode('utf-8')
            pipe_conn.send_bytes(error_response)
        except:
            pass

import io

def migrate():
    results_dir = "results"
    if not os.path.exists(results_dir):
        print("No results directory found.")
        return

    files = glob.glob(os.path.join(results_dir, "optimizer_state_*.pkl"))
    if not files:
        print("No pickle state files found.")
        return

    print(f"Found {len(files)} state files to migrate.")

    for pkl_file in files:
        print(f"Migrating {pkl_file}...")
        
        # Read content in parent to avoid file access in child
        try:
            file_size = os.path.getsize(pkl_file)
            if file_size > 50 * 1024 * 1024:
                print(f"Skipping {pkl_file}: Too large ({file_size} bytes)")
                continue
                
            with open(pkl_file, 'rb') as f:
                content = f.read()
        except Exception as e:
            print(f"Failed to read {pkl_file}: {e}")
            continue

        # Spawn child
        parent_conn, child_conn = multiprocessing.Pipe()
        p = multiprocessing.Process(target=safe_load_process, args=(content, child_conn))
        p.start()
        
        # Wait with timeout
        p.join(timeout=10)
        
        if p.is_alive():
            print("Timeout! Killing process...")
            p.terminate()
            p.join()
            print("Migration failed for this file (Timeout).")
            continue
            
        if parent_conn.poll(10): # Poll with timeout
            try:
                # Secure IPC: Receive bytes and decode JSON
                # This prevents implicit unpickling of malicious data
                result_bytes = parent_conn.recv_bytes()
                result = json.loads(result_bytes.decode('utf-8'))
                
                if result.get('success'):
                    data = result['data']
                    json_file = pkl_file.replace('.pkl', '.json')
                    
                    # Write JSON
                    try:
                        with open(json_file, 'w') as f:
                            json.dump(data, f, indent=2)
                        print(f"Successfully migrated to {json_file}")
                    except Exception as e:
                        print(f"Failed to write JSON: {e}")
                else:
                    print(f"Migration failed: {result.get('error')}")
            except Exception as e:
                print(f"Migration failed (IPC Error): {e}")
        else:
            print("Migration failed: No response from child (Timeout/Poll).")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    migrate()
