import h5py , os, torch
from tqdm import tqdm
from ..tokenizer import SimpleTokenizer
import re


def make_h5(pt_data_folder, tokenizer:SimpleTokenizer, data_name=None,destination_folder = None, dtype='int'):
    """
        Make h5 dataset from a folder of pt files.

        Args:
            pt_data_folder (str): Folder containing pt files
            tokenizer (SimpleTokenizer): Tokenizer to use to visualize data, not used for the h5 file itself.
            data_name (str, optional): Name of the dataset. Defaults to None, in which case the name of the folder is used.
            destination_folder (str, optional): Folder to save the h5 file. Defaults to 'h5data', in which case the current folder is used.
    """

    if(destination_folder is None):
        destination_folder= 'h5data'

    if(data_name is None):	
        tarname = os.path.join(destination_folder,f'{os.path.basename(pt_data_folder)}.h5')
    else :
        tarname = os.path.join(destination_folder,f'{data_name}.h5')
    os.makedirs(os.path.dirname(tarname),exist_ok=True)


    if(os.path.isdir(pt_data_folder)):
        # extract_file = os.listdir(pt_data_folder)[0]
        # tensor = torch.load(os.path.join(pt_data_folder,extract_file))
        with h5py.File(tarname, 'w') as f:
            dset = f.create_dataset("tokens", shape=(0,),maxshape=(None,), chunks=(512,), dtype=dtype)  # note the maxshape parameter
            print('Created empty dataset')    
            dset_index=0
            file_list = os.listdir(pt_data_folder)
            file_list = sorted(file_list, key=lambda x: int(re.search(r'\d+', x).group()))
            print('Sorted file_list : \n', file_list)
            for file in file_list:
                current_index = 0
                print('Treating file : ', file)
                if os.path.splitext(file)[1]=='.pt':
                    pt_file = os.path.join(pt_data_folder,file)
                    tensor = torch.load(pt_file,map_location=torch.device('cpu')) # (length)
                    tot_length = tensor.shape[0]
                    print(f'Tensor shape: {tensor.shape}')

                    print('snippet', tokenizer.detokenize(tensor[:30]))
                    loadbar = tqdm(total=tot_length)
                    while current_index < tot_length:
                        phrase_length = min(256*1024*1024, tot_length - current_index)
                        # Resize the dataset to accommodate the new data
                        dset.resize((dset_index + phrase_length,))
                        
                        # Add the new data to the dataset
                        dset[dset_index:dset_index+phrase_length] = tensor[current_index:current_index+phrase_length].numpy()
                        
                        # Update the current ind
                        current_index += phrase_length
                        dset_index += phrase_length
                        loadbar.update(phrase_length)
                    loadbar.close()
    else :
        raise ValueError(f'{pt_data_folder} not found')


