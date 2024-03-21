import json, os, sys

sys.path.append('../..')

from modules.util import DATA_FOLDER

def modifparam(blueprint,anim_name=None):
    """
        Modifies blueprint and saves it to anim_name. If anim_name is None, it overwrites the blueprint.
    """
    if blueprint.endswith('.json'):
        with open(blueprint, 'r') as f:
            data = json.load(f)
            data['training_params']["dataset_folder"] = os.path.join(DATA_FOLDER,f"{anim_name}.h5")
            with open(anim_name+'.json', 'w') as f:
                json.dump(data, f, indent=4)


    return "JSON files created successfully."

if __name__=='__main__':
    for file in os.listdir(DATA_FOLDER):
        if file.endswith('.fa'):
            anim_name = file.split('.')[0]
            modifparam('blueprint.json',anim_name=anim_name)