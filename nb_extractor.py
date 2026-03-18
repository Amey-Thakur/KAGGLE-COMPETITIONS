import json

with open('D:/Kaggle/petals-to-the-metal-swin-transformer-tpu.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('extracted_nb.txt', 'w', encoding='utf-8') as out:
    for i, cell in enumerate(nb['cells']):
        out.write(f'--- CELL {i} ({cell["cell_type"]}) ---\n')
        source = "".join(cell.get("source", []))
        out.write(source + '\n\n')
