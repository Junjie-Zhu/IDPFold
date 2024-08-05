import os

# Get current absolute path
current_path = os.getcwd()

# content of .env file
content_list = {'CACHE_DIR': '.cache',
                'TRAIN_DATA': 'data/pdb',
                'EMBEDDING': 'data/embeddings',
                'TEST_DATA': 'data/test_pdb',
                }

# Write .env file
with open('./.env', 'w') as f:
    for keys, values in content_list.items():
        target_path = os.path.join(current_path, values)

        if not os.path.isdir(target_path):
            os.mkdir(target_path)

        f.write(f'{keys}=\"{target_path}\"\n')
