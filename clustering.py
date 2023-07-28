import torch
import numpy as np
import matplotlib.pyplot as plt
import json

from datasets import load_dataset
from kmeans_pytorch import kmeans, kmeans_predict
from sentence_transformers import SentenceTransformer

import fire

def cluster(
    data_path: str,
    model: str = 'jhgan/ko-sroberta-multitask',
    output_dir: str = './',
    split: str = 'train',
    num_clusters: int = 4,
):
    dataset = load_dataset(data_path, split=split)
    device = torch.device('cuda')

    embedding_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    embeddings = torch.tensor(embedding_model.encode(dataset['instruction']))
    
    cluster_ids_x, cluster_centers = kmeans(
        X=embeddings, num_clusters=num_clusters, distance='euclidean', device=device
    )
    
    cluster_ids = []
    for x in cluster_ids_x:
        cluster_ids.append(int(x))
    
    clustered_dataset = []
    clustered_dataset_1, clustered_dataset_2, clustered_dataset_3, clustered_dataset_4 = [], [] ,[] ,[]
    for idx, inst, output in zip(cluster_ids, dataset['instruction'], dataset['output']):
        clustered_dataset.append(
            {
                'instruction': inst,
                'output': output,
                'cluster_id': idx
            }
        )
        if idx == 0:
            clustered_dataset_1.append(
                {
                    'instruction': inst,
                    'output': output,
                    'cluster_id': idx
                }
            )
        if idx == 1:
            clustered_dataset_2.append(
            {
                'instruction': inst,
                'output': output,
                'cluster_id': idx
            }
        )
        if idx == 2:
            clustered_dataset_3.append(
            {
                'instruction': inst,
                'output': output,
                'cluster_id': idx
            }
        )
        if idx == 3:
            clustered_dataset_4.append(
            {
                'instruction': inst,
                'output': output,
                'cluster_id': idx
            }
        )
            
    with open('./datasets/EvolInstruct_clustered.json', 'w', encoding='utf-8') as f:
        json.dump(clustered_dataset, f, ensure_ascii=False, indent='\t')
    with open('./datasets/c1.json', 'w', encoding='utf-8') as f:
        json.dump(clustered_dataset_1, f, ensure_ascii=False, indent='\t')
    with open('./datasets/c2.json', 'w', encoding='utf-8') as f:
        json.dump(clustered_dataset_2, f, ensure_ascii=False, indent='\t')
    with open('./datasets/c3.json', 'w', encoding='utf-8') as f:
        json.dump(clustered_dataset_3, f, ensure_ascii=False, indent='\t')
    with open('./datasets/c4.json', 'w', encoding='utf-8') as f:
        json.dump(clustered_dataset_4, f, ensure_ascii=False, indent='\t')

    torch.save(cluster_centers, './datasets/centers.pt')

if __name__ == '__main__':
    np.random.seed(42)
    fire.Fire(cluster)
