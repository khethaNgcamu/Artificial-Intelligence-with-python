from pyswip import Prolog

# Initialize Prolog
prolog = Prolog()

# Load the Prolog file
prolog.consult("sudoku.pl")

# Example Sudoku puzzle (0 represents blank cells)
puzzle = [5,3,0,0,7,0,0,0,0,
          6,0,0,1,9,5,0,0,0,
          0,9,8,0,0,0,0,6,0,
          8,0,0,0,6,0,0,0,3,
          4,0,0,8,0,3,0,0,1,
          7,0,0,0,2,0,0,0,6,
          0,6,0,0,0,0,2,8,0,
          0,0,0,4,1,9,0,0,5,
          0,0,0,0,8,0,0,7,9]

# Solve the puzzle
query = list(prolog.query(f"sudoku({puzzle}, Solution)"))

# Output the solution
if query:
    print("Solved Sudoku Grid:", query[0]['Solution'])
else:
    print("No solution found.")
