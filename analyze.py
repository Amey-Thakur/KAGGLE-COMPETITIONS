import re

with open('notebook_summary.txt', 'r', encoding='utf-8') as f:
    text = f.read()

notebooks = text.split('================================================================================')

for nb in notebooks:
    if not nb.strip(): continue
    lines = nb.split('\n')
    title = 'Unknown'
    for line in lines:
        if line.startswith('FILE:'):
            title = line
            break

    if title == 'Unknown': continue
    
    print('******************************************')
    print(title)
    
    # search for parameters
    epochs = re.search(r'EPOCHS\s*=\s*(\d+)', nb, re.IGNORECASE)
    img_size = re.search(r'IMAGE_SIZE\s*=\s*\[?(\d+)[,\s]+(\d+)\]?', nb, re.IGNORECASE)
    batch_size = re.search(r'BATCH_SIZE\s*=\s*(.*)', nb, re.IGNORECASE)
    models = set(re.findall(r'(EfficientNetB\d|DenseNet\d+|SwinTransformer|VGG\d|ResNet\d+)', nb, re.IGNORECASE))
    
    # checks
    has_tpu = 'TPUClusterResolver' in nb or 'TPU' in nb
    has_external_data = 'tfrecords-jpeg' in nb or 'external' in nb.lower()
    
    print('TPU Used: ' + str(has_tpu))
    print('External Data: ' + str(has_external_data))
    print('Epochs: ' + (epochs.group(1) if epochs else 'Unknown'))
    if img_size:
        print('Image Size: ' + img_size.group(1) + 'x' + img_size.group(2))
    else:
        print('Image Size: Unknown')
    print('Batch Size code: ' + (batch_size.group(1) if batch_size else 'Unknown'))
    print('Models mentioned: ' + str(models))
    print('******************************************\n')
