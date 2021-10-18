import faiss
from app import retrieval

from utils.feature_indexing import get_arg
from utils.feature_indexing import IndexingRetrievalModel
from configs.config import init_config


my_cfg = init_config()

if __name__ == '__main__':
    
    retrieval_model = IndexingRetrievalModel(get_arg(), my_cfg)
    query = input("Query: ")
    result = retrieval_model.retrieval(query)

    print(result)