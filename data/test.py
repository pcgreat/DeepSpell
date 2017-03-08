with open("input.txt", "w") as g:
    with open("input_old.txt", "r") as f:
        for line in f:
            g.write(line.lower())