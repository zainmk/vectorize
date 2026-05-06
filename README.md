# [vectorize](https://vector1ze.vercel.app/)
'vectorize' data to enable semantic search via sentence-transformers

#### purpose
the purpose of this application is to explore the use of 'vector databases' and [RAG based semantic search](https://www.youtube.com/watch?v=JB2P5Gk23VI). Transforming multimodal data into vector embeddings, allows for much more accurate and possible 'semantic search'. We can use this for RAG tools, where a specific documented reference is required. This can be useful to cater AI outputs as per controlled 'input' documentations.

#### tools & algs
- BM25 (Best-Matching 25): Commonly currently used to query search results against a database. Tokenizes the 'strings' (title, description) / documents as per term frequency, inverse document frequency, and document length normalization. This search alg. is limited by being incapable of producing results that do not include the original search query 'strings'.
- SemanticSearch (via ChromaDB): Using chromaDB, we vectorize the original data - then search directly on it 'semantically'. This method produces vectors based on the context of the data (title, description), then when queried, returns the closest document vector to the search vector. This allows for results that do not include the original search literal's but are still relevant to the user.
- RAG (Retrieval-Augmented Generation): A process of facilitating a LLM response w. specific 'knowledge', is commonly performed using vector databases. The query input from the user is similarily vector-embedded, then matched against the vector database to identify similar vectors and return a related response.


#### [model2vec](https://pypi.org/project/model2vec/0.3.7/)
the `sentence-transformers` lib is typically used to model the data and encode the search queries. To deploy via vercel however, the total deployed size is limited to 500MB ~ sentence-transformers requires PyTorch which exceeds that file size.

We use a lightweight alternative, `model2vec` and instead use a generated pretrained model (less 'contextual' performance between the input's themselves) that is saved locally, but is smaller and faster. On server initialization, we use the pretrained model to encode the dataset of 30 movies into vector embeddings, and then do the same to the search queries to evaluate the 'closest' ones. 

Upgrading to sentence-transformers and using 'contextual' embedding models instead of static, is worth the overhead when...
- queries are lingusiticly complex
- relevancy between input data is required (domain specificity)
- large amount of input data (this ex. only uses a dataset of 30 movies)
- cross-lingual search
