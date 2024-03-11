import os


def merge_text(text_folder, output_file_path, chunk_size=1024*1024):
    """
    Merges multiple large files into a single file.

    Args :
        text_folder (str) : Folder containing the text files to merge.
        output_file_path (str) : Path to save the merged file.
        chunk_size (int, optional) : Size of the chunks to read and write. Defaults to 1024*1024.
    """

    with open(output_file_path, 'wb') as outfile:
        for file_path in os.listdir(text_folder):
            if(file_path.endswith('.txt')):
                with open(os.path.join(text_folder,file_path), 'rb') as infile:
                    while True:
                        chunk = infile.read(chunk_size)
                        if not chunk:
                            break
                        outfile.write(chunk)
