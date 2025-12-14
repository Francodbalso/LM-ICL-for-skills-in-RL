import re

def get_best_move(state_text):
    """
    Parses the game state and returns the optimal next move.
    """
    # 1. Parse the board state
    towers = parse_board(state_text)
    
    # 2. Identify the number of disks and their locations
    # We flatten the lists to find the max disk number 'N'
    all_disks = [d for tower in towers.values() for d in tower]
    if not all_disks:
        return "No disks found on board."
    
    n_disks = max(all_disks)
    
    # Map disk values to their current tower (e.g., {1: 'A', 2: 'B', ...})
    disk_locations = {}
    for peg, disks in towers.items():
        for disk in disks:
            disk_locations[disk] = peg

    # 3. Determine the optimal move
    # We want to move all N disks to the target peg (usually 'C').
    # We iterate from the largest disk down. The first time we find a disk
    # that isn't where it needs to be (based on the plan), that mismatch 
    # dictates the next immediate move.
    
    target_peg = 'C' 
    move_to_make = None

    for disk in range(n_disks, 0, -1):
        current_peg = disk_locations.get(disk)
        
        if current_peg != target_peg:
            # This disk is NOT where it needs to be.
            # Therefore, we MUST move this disk to the target_peg.
            # But we can only do that if all smaller disks are out of the way.
            
            # This becomes our candidate move:
            move_to_make = (current_peg, target_peg)
            
            # For the NEXT smallest disk (disk - 1), it must be on the 
            # AUXILIARY peg to clear the way.
            # The auxiliary peg is the one that is neither source nor target.
            # We assume pegs are 'A', 'B', 'C'.
            all_pegs = {'A', 'B', 'C'}
            aux_peg = list(all_pegs - {current_peg, target_peg})[0]
            
            # Update the target for the next iteration (smaller disk)
            target_peg = aux_peg
        else:
            # This disk is already correct for the current plan.
            # We don't need to move it.
            # The target for the disk above it (disk-1) remains the same
            # because we want to stack it on top of this one.
            pass

    # 4. Format the output
    if move_to_make:
        return f"[{move_to_make[0]} {move_to_make[1]}]"
    else:
        return "Solved"

def parse_board(text):
    """
    Extracts the tower lists from the raw text.
    Expected format in text:
    A: [3, 2, 1]
    B: []
    C: []
    """
    state = {'A': [], 'B': [], 'C': []}
    
    # Regex to capture "Letter: [numbers]"
    # It looks for A, B, or C, followed by a colon, brackets, and content
    matches = re.findall(r'([ABC]):\s*\[(.*?)\]', text)
    
    for peg, content in matches:
        if content.strip():
            # Convert "3, 2, 1" string to list of ints [3, 2, 1]
            disks = [int(x.strip()) for x in content.split(',')]
            state[peg] = disks
        else:
            state[peg] = []
            
    return state