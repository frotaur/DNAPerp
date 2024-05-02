import os, shutil, json

list = [9,10,11,12,13,14,15,16]
for i in list:
    # Copy human310_1.json in human310_i.json
    file = json.load(open('human310_b_1.json'))
    file['training_params']['seed'] = 300+i
    with open('human310_b_'+str(i)+'.json', 'w') as f:
        json.dump(file, f, indent=4)