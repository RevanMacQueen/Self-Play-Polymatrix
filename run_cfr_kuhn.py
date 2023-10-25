import os

for i in range(0, 30):
    os.system('python3 run_cfr.py --alg cfr --game kuhn_poker --save_dir cfr/cfr_kuhn/{} --seed {}'.format(i, i))