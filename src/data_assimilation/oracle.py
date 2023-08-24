from array import array
import os 
from pathlib import Path
import pdb
import pickle
import numpy as np 
import oci 
from multiprocessing import Process 
from multiprocessing import Semaphore 
import ocifs
import torch
import yaml

# Number of max processes allowed at a time 
concurrency= 5 
sema = Semaphore(concurrency) 
# The root directory path, Replace with your path 
p = Path('/Users/nikolajmucke/cwi/latent-time-stepping/data/training_data/state') 
# The Compartment OCID 
compartment_id = "ocid1.tenancy.oc1..aaaaaaaaeadwopiezanqrr5ybd3w6wwmzzqavceibijgl46upfkgyonu7otq"
# The Bucket name where we will upload 
bucket_name = "bucket-20230222-1753" 

def upload_to_object_storage(
    source_path: str,
    destination_path:str, 
    object_storage_client,
    namespace
): 
    
    with open(source_path, "rb") as in_file: 
        object_storage_client.put_numpy_object(namespace,bucket_name,destination_path,in_file) 


def download_from_object_storage(
    source_path,
    destination_path,
    object_storage_client,
    namespace
):
    
    get_obj = object_storage_client.get_numpy_object(namespace,bucket_name,source_path,)

    with open(destination_path, 'wb') as f:
        for chunk in get_obj.data.raw.stream(2, decode_content=False):
            f.write(chunk)

class ObjectStorageClientWrapper:
    def __init__(self, bucket_name):

        config = oci.config.from_file() 
        self.object_storage_client = oci.object_storage.ObjectStorageClient(config)

        self.namespace = self.object_storage_client.get_namespace().data 

        self.bucket_name = bucket_name

        self.fs = ocifs.OCIFileSystem(config)

    def put_numpy_object(self, data, destination_path): #, source_path):

        with self.fs.open(f'{self.bucket_name}@{self.namespace}/{destination_path}', 'wb') as f:
            if destination_path[-3:] == 'npz':
                np.savez_compressed(f, data=data)
            else:
                np.save(f, data)

        '''
        upload_to_object_storage(
            source_path=source_path,
            destination_path=destination_path,
            object_storage_client=self.object_storage_client,
            namespace=self.namespace
        )
        '''

    def get_numpy_object(self, source_path):

        with self.fs.open(f'{self.bucket_name}@{self.namespace}/{source_path}', 'rb') as f:
            data = np.load(f)
            if source_path[-3:] == 'npz':
                data=data['data']

        return data
    
    def put_model(self, source_path, destination_path, with_config=True):

        ##### State dicts #####
        # Load the model
        model = torch.load(f'{source_path}/model.pt')

        # Upload the model to object storage
        with self.fs.open(f'{self.bucket_name}@{self.namespace}/{destination_path}/model.pt', 'wb') as f:
            torch.save(model, f)

        ##### Config #####    
        # Load the config   
        with open(f'{source_path}/config.yml') as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)

        # Upload the config to object storage
        if with_config:
            with self.fs.open(f'{self.bucket_name}@{self.namespace}/{destination_path}/config.yml', 'w') as f:
                yaml.dump(config, f)
    
    def get_model(self, source_path, with_config=True, device='cpu'):
        
        # Load the model state dicts from object storage
        with self.fs.open(f'{self.bucket_name}@{self.namespace}/{source_path}/model.pt', 'rb') as f:
            state_dict = torch.load(f, map_location=device)

        # Load the config from object storage
        if with_config:
            with self.fs.open(f'{self.bucket_name}@{self.namespace}/{source_path}/config.yml', 'rb') as f:
                config = yaml.load(f, Loader=yaml.SafeLoader)

            return state_dict, config  
        
        return state_dict
    
    def put_preprocessor(self, source_path, destination_path):

        # Load the model
        #preprocessor = torch.load(source_path)
        with open(source_path, 'rb') as f:
            preprocessor = pickle.load(f)

        # Upload the model to object storage
        with self.fs.open(f'{self.bucket_name}@{self.namespace}/{destination_path}', 'wb') as f:
            pickle.dump(preprocessor, f)
            #torch.save(preprocessor, f)

    def get_preprocessor(self, source_path):
        
        # Load the preprocessor from object storage
        with self.fs.open(f'{self.bucket_name}@{self.namespace}/{source_path}', 'rb') as f:
            preprocessor = pickle.load(f)

            #preprocessor = torch.load(f)

        return preprocessor
    




if __name__ == '__main__': 
    config = oci.config.from_file() 
    object_storage_client = oci.object_storage.ObjectStorageClient(config) 
    namespace = object_storage_client.get_namespace().data 
    proc_list: array = []

    object_loader = ObjectStorageClientWrapper(bucket_name)

    for i in range(1,5):
        object_loader.put_numpy_object(
            destination_path=f'state/sample_{i}.npy',
            source_path=f'/Users/nikolajmucke/cwi/latent-time-stepping/data/training_data/state/sample_{i}.npy'
        )

    for i in range(1,5):
        object_loader.get_numpy_object(
            source_path=f'state/sample_{i}.npy',
            destination_path=f'/Users/nikolajmucke/cwi/latent-time-stepping/sample_{i}.npy'
        )