import os

if __name__ == '__main__':
    batch_size = 80
    batch_size_eval = 2 * batch_size
    epochs = 30
    lr = 2e-4
    mode = ['original', 'grouping', 'ekman'][0]
    gpu = '0'   # which gpu to use
    n_workers = 4
    threshold = 0.5

    command = f'CUDA_VISIBLE_DEVICES={gpu} python -u train.py ' \
              f'--batch_size={batch_size} ' \
              f'--batch_size_eval={batch_size_eval} ' \
              f'--epochs={epochs} ' \
              f'--lr={lr} ' \
              f'--gpu={gpu} ' \
              f'--n_workers={n_workers} ' \
              f'--threshold={threshold} ' \
              f'--mode={mode}'

    print(command)
    print()
    os.system(command)

    script = f"""#!/bin/bash
    #SBATCH --partition=gpu
    #SBATCH --nodes=1
    #SBATCH --cpus-per-task=4
    #SBATCH --gres=gpu:v100:1
    #SBATCH --mem=16GB
    #SBATCH --time=2:00:00

    {command}
            """
    with open('run.sh', 'w') as f:
        f.write(script)
    os.system('sbatch run.sh')