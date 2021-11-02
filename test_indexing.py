from utils.feature_indexing import get_arg
from utils.feature_indexing import IndexingRetrievalModel
from configs.config import init_config

my_cfg = init_config()

retrieval_model = IndexingRetrievalModel(get_arg(), my_cfg)

query = input(str)
result = retrieval_model.retrieval(query)
print(result[0])