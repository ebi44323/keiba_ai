with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    # lines 601 to 647 are index 600 to 646.
    # Actually let's find the exact block.
    if 599 < i < 648:
        if line.startswith('                        '):
            line = line[8:] # remove 8 spaces
        elif line.startswith('                            '):
            line = line[8:] 
        elif line.strip() == '':
            pass
        else:
            # Maybe some lines have fewer spaces, just remove up to 8
            spaces = len(line) - len(line.lstrip())
            if spaces >= 8:
                line = line[8:]
    new_lines.append(line)

with open('app.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("Dedented.")
