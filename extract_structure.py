import ast

with open('app.py', 'r', encoding='utf-8') as f:
    source = f.read()

tree = ast.parse(source)
for node in tree.body:
    if isinstance(node, ast.FunctionDef):
        print(f"Function: {node.name} (Line: {node.lineno} to {node.end_lineno})")
    elif isinstance(node, ast.ClassDef):
        print(f"Class: {node.name} (Line: {node.lineno} to {node.end_lineno})")
    elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
        if node.targets[0].id.isupper() or node.targets[0].id.startswith('_'):
            print(f"Global: {node.targets[0].id} (Line: {node.lineno})")
