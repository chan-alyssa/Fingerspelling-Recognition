import h5py             

class ReadH5:
    def __init__(self, h5_path) -> None:
        self.h5_path = h5_path  
        
    def read_all_datasets(self):
        all_data = []
        def read_all_datasets_(h5py_file, path=''):
            for key in h5py_file.keys():
                item = h5py_file[key]
                new_path = f'{path}/{key}'
                if isinstance(item, h5py.Dataset):
                    all_data.append(item[...])
                elif isinstance(item, h5py.Group):
                    read_all_datasets_(item, new_path)
                    
        with h5py.File(self.h5_path, 'r') as f:
            read_all_datasets_(f)
        return all_data
    
    def read_sequence(self, sequence_name):
        with h5py.File(self.h5_path, 'r') as f:
            return f[sequence_name][...]
    
    def read_sequence_slice(self, sequence_name, slice_):
        with h5py.File(self.h5_path, 'r') as f:
            return f[sequence_name][slice_]
        
    def get_all_sequence_names(self):
        all_names = []
        def read_all_names_(h5py_file, path=''):
            for key in h5py_file.keys():
                item = h5py_file[key]
                new_path = f'{path}/{key}'
                if isinstance(item, h5py.Dataset):
                    all_names.append(new_path)
                elif isinstance(item, h5py.Group):
                    read_all_names_(item, new_path)
                    
        with h5py.File(self.h5_path, 'r') as f:
            read_all_names_(f)
        return all_names
    
if __name__ == "__main__":
    h = "/home/zador/workspace/projects/casa_plus_cvpr24/npyrecords/h2o_dino.h5"
    print(ReadH5(h).get_all_sequence_names())