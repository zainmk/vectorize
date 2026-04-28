# vectorize
'vectorize' data to enable semantic search via LLM's for RAG

#### purpose
the purpose of this application is to explore the use of 'vector databases'. Transforming multimodal data into vector embeddings, allows for much more accurate and possible 'semantic search'. We can use this for RAG tools, where a specific documented reference is required. This can be useful to cater AI outputs as per controlled 'input' documentations.

#### tools & algs
- BM25 (Best-Matching 25): Commonly currently used to query search results against a database. Tokenizes the 'strings' (title, description) / documents as per term frequency, inverse document frequency, and document length normalization. This search alg. is limited by being incapable of producing results that do not include the original search query 'strings'.
- SemanticSearch (via ChromaDB): Using chromaDB, we vectorize the original data - then search directly on it 'semantically'. This method produces vectors based on the context of the data (title, description), then when queried, returns the closest document vector to the search vector. This allows for results that do not include the original search literal's but are still relevant to the user.
- RAG (Retrieval-Augmented Generation): A process of facilitating a LLM response w. specific 'knowledge', is commonly performed using vector databases. The query input from the user is similarily vector-embedded, then matched against the vector database to identify similar vectors and return a related response.

#### vercel vs. render
the deployment tool used here was 'render'
vercel's 500MB serverless lambda function impl. restricts installing chromadb, sentence-transformers, etc... (large libraries required for model analysis). therefore going w. render for it's free tier offerings and more robust backend support for deployment.

<img width="2299" height="782" alt="image" src="https://github.com/user-attachments/assets/54937c1d-a2d7-4bd7-8d7c-73f2ab15609d" />
