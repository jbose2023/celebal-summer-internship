def lower_triangle(rows):
    for i in range(1, rows + 1):
        print('*' * i)

def upper_triangle(rows):
    for i in range(rows, 0, -1):
        print(' ' * (rows - i) + '*' * i)

def pyramid(rows):
    for i in range(1, rows + 1):
        spaces = ' ' * (rows - i)
        stars = '*' * (2 * i - 1)
        print(spaces + stars + spaces)

# Set the number of rows
rows = 5

print("Lower Triangle:")
lower_triangle(rows)

print("\nUpper Triangle:")
upper_triangle(rows)

print("\nPyramid:")
pyramid(rows)
