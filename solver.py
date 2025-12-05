import time
import argparse

class ScrewTetracubeSolver:
    def __init__(self, chirality='right'):
        # 1. Define the Unit Screw Tetracube
        self.chirality = chirality
        if chirality == 'right':
            # Right-handed
            self.base_piece_coords = [
                (0, 0, 0),
                (1, 0, 0),
                (1, 1, 0),
                (1, 1, 1)
            ]
        else:
            # Left-handed (Mirror of right-handed)
            # Mirroring across X axis
            self.base_piece_coords = [
                (0, 0, 0),
                (-1, 0, 0),
                (-1, 1, 0),
                (-1, 1, 1)
            ]
        
        print(f"Initializing {chirality}-handed solver...")
        
        # 2. Generate all 24 Rotations
        self.orientations = self._generate_orientations()
        
        # 3. Define the Target Shape (3x Scale)
        # The target is made of 4 blocks of size 3x3x3
        # Positions: Origin, Right, Back, Top
        # Scale factor = 3
        self.target_voxels = set()
        
        block_origins = [
            (0, 0, 0), 
            (3, 0, 0), 
            (3, 3, 0), 
            (3, 3, 3)
        ]
        
        block_size = 3
        
        print("Generating 3x Target Shape...")
        for ox, oy, oz in block_origins:
            for dx in range(block_size):
                for dy in range(block_size):
                    for dz in range(block_size):
                        coord = (ox + dx, oy + dy, oz + dz)
                        self.target_voxels.add(coord)
        
        self.solutions = []
        self.start_time = 0

    def _generate_orientations(self):
        """Generates all 24 unique rotations of the base piece."""
        def rotate_x(p): return (p[0], -p[2], p[1])
        def rotate_y(p): return (p[2], p[1], -p[0])
        def rotate_z(p): return (-p[1], p[0], p[2])

        unique_shapes = set()
        
        # BFS to find all 24 orientations
        seen_states = set()
        queue = [tuple(sorted(self.base_piece_coords))]
        seen_states.add(queue[0])
        
        final_orientations = []

        while queue:
            current_shape = queue.pop(0)
            
            # Normalize
            min_x = min(c[0] for c in current_shape)
            min_y = min(c[1] for c in current_shape)
            min_z = min(c[2] for c in current_shape)
            
            norm_shape = tuple(sorted([(c[0]-min_x, c[1]-min_y, c[2]-min_z) for c in current_shape]))
            
            if norm_shape not in unique_shapes:
                unique_shapes.add(norm_shape)
                final_orientations.append(norm_shape)
            
            for rot_func in [rotate_x, rotate_y, rotate_z]:
                next_shape = tuple(sorted([rot_func(c) for c in current_shape]))
                if next_shape not in seen_states:
                    seen_states.add(next_shape)
                    queue.append(next_shape)
                    
        return final_orientations

    def solve(self):
        target_vol = len(self.target_voxels)
        pieces_needed = target_vol // 4
        
        print(f"Target Volume: {target_vol} units.")
        print(f"Pieces needed: {pieces_needed}.")
        print(f"Unique Piece Orientations: {len(self.orientations)}")
        print(f"Starting solver for 3x scale ({self.chirality}-handed)...")
        print("Note: All solutions for 3x are inherently non-trivial.")
        
        self.start_time = time.time()
        
        # Grid state
        grid = {v: None for v in self.target_voxels}
        
        # Sort target list for deterministic filling (Bottom-up heuristic)
        # Z is primary sort key to keep the footprint compact
        target_list = sorted(list(self.target_voxels), key=lambda t: (t[2], t[1], t[0]))
        
        self._backtrack(grid, target_list, 1)
        
        print(f"\nSearch complete.")
        print(f"Total Solutions Found: {len(self.solutions)}")

    def _backtrack(self, grid, target_list, piece_id):
        # Base case
        if piece_id > 27:
            self._record_solution(grid)
            return

        # Optimization: Check if current state has isolated holes
        # (Skip for now to keep code simple, but valuable for 27-piece depth)

        # Find first empty slot
        first_empty = None
        for voxel in target_list:
            if grid[voxel] is None:
                first_empty = voxel
                break
        
        if first_empty is None:
            return

        fx, fy, fz = first_empty

        for orientation in self.orientations:
            # Anchor checking
            for atom_index, (ax, ay, az) in enumerate(orientation):
                shift_x = fx - ax
                shift_y = fy - ay
                shift_z = fz - az
                
                can_place = True
                placement_coords = []
                
                for (cx, cy, cz) in orientation:
                    tx, ty, tz = cx + shift_x, cy + shift_y, cz + shift_z
                    target_coord = (tx, ty, tz)
                    
                    if target_coord not in grid or grid[target_coord] is not None:
                        can_place = False
                        break
                        
                    placement_coords.append(target_coord)
                
                if can_place:
                    # Place
                    for coord in placement_coords:
                        grid[coord] = piece_id
                    
                    # Recurse
                    self._backtrack(grid, target_list, piece_id + 1)
                    
                    # Remove (Backtrack)
                    for coord in placement_coords:
                        grid[coord] = None
                    
                    # Optimization:
                    # If we successfully placed a piece at 'first_empty' using this specific orientation,
                    # but it led to no solution, do we need to try shifting this orientation?
                    # No, because 'first_empty' MUST be filled.
                    # However, we are iterating through atoms, so we are effectively shifting.
                    # This logic is correct.

    def _record_solution(self, grid):
        self.solutions.append(grid.copy())
        
        count = len(self.solutions)
        if count == 1:
            print(f"First solution found in {time.time() - self.start_time:.2f} seconds!")
            self._print_visual(grid)
        elif count % 100 == 0:
            print(f"Found {count} solutions...", end='\r')

    def _print_visual(self, grid):
        print(f"\n--- VISUALIZATION OF 3X SOLUTION #1 ({self.chirality.upper()}) ---")
        print("Slices through Z axis (Bottom to Top)")
        
        # 3x shape extends to 6 in Z, 6 in X, 6 in Y
        max_dim = 6 
        
        for z in range(max_dim):
            print(f"\nLayer Z={z}")
            print("   " + " ".join([str(x) for x in range(max_dim)]))
            print("  +" + "-" * (max_dim * 2))
            
            for y in range(max_dim):
                row_str = f"{y} |"
                for x in range(max_dim):
                    val = grid.get((x, y, z))
                    if val is None:
                        if (x,y,z) in self.target_voxels:
                            row_str += " ."
                        else:
                            row_str += "  "
                    else:
                        # Print piece ID (mod 10 for single digit spacing)
                        row_str += f"{val%10} "
                print(row_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve the Screw Tetracube puzzle (3x scale).")
    parser.add_argument('--chirality', choices=['right', 'left'], default='right', 
                        help="Chirality of the screw tetracube (right or left). Default is right.")
    
    args = parser.parse_args()
    
    solver = ScrewTetracubeSolver(chirality=args.chirality)
    solver.solve()