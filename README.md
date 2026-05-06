# [vectorize](https://vector1ze.vercel.app/)
'vectorize' data to enable semantic search via sentence-transformers

#### purpose
the purpose of this application is to explore the use of 'vector databases'. Transforming multimodal data into vector embeddings, allows for much more accurate and possible contextual based 'semantic search'. This project compares these search methods (semantic vs. BM25) and how each still proves its use side by side. These tools are used w.in the RAG pipeline.

#### tools & algs
- BM25 (Best-Matching 25): Commonly currently used to query search results against a database. Tokenizes the 'strings' (title, description) / documents as per term frequency, inverse document frequency, and document length normalization. This search alg. is limited by being incapable of producing results that do not include the original search query 'strings'.
- SemanticSearch (via ChromaDB): Using chromaDB, we vectorize the original data - then search directly on it 'semantically'. This method produces vectors based on the context of the data (title, description), then when queried, returns the closest document vector to the search vector. This allows for results that do not include the original search literal's but are still relevant to the user.
- RAG (Retrieval-Augmented Generation): A process of facilitating a LLM response w. specific 'knowledge', is commonly performed using vector databases. The query input from the user is similarily vector-embedded, then matched against the vector database to identify similar vectors and return a related response.


#### [model2vec](https://pypi.org/project/model2vec/0.3.7/)
the `sentence-transformers` lib is typically used to embed the data and query. To deploy via Vercel however, the total deployed size is limited to 500MB ~ sentence-transformers requires PyTorch which exceeds that file size. To reduce the total size of the application, we use a distilled static model, locally saved, to encode our dataset. The search queries are instead tokenized, mapped to a static embedding table, then scores averaged out to find the best answer. This  allows for a more lightweight application, but restricts the response to not have 'self attention' and vectorize the query itself for semantic understanding (ex. asking for 'a movie that has nothing to do with space', will not work as the 'in-query' understanding of 'not-space' won't be contextually picked up)

Upgrading to sentence-transformers and using 'contextual' embedding models instead of static, is worth the overhead when...
- queries are lingusiticly complex
- relevancy between input data is required (domain specificity)
- large amount of input data (this ex. only uses a dataset of 30 movies)
- cross-lingual search
