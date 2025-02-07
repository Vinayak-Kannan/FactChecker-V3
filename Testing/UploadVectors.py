from dotenv import load_dotenv
from pinecone import Pinecone
from tqdm import tqdm
import re
import os
import chromadb

load_dotenv()


pc = Pinecone(api_key=os.getenv("PINECONE_KEY"))

index = pc.Index("factchecker")
chroma_client = chromadb.PersistentClient(path="/Users/vinayakkannan/Desktop/Projects/FactChecker/FactChecker/Clustering/Clustering/Chroma")
col = chroma_client.get_collection('climate_claims_embeddings_unchanged')
reduced_collection = col.get(include=['embeddings', 'documents', 'metadatas'])

embeddings = reduced_collection['embeddings']
documents = reduced_collection['documents']
metadatas = reduced_collection['metadatas']

data = []
embeddings = embeddings[13040:]
print(len(embeddings))
for i, embedding in tqdm(enumerate(embeddings)):
    id = re.sub(r'[^\x00-\x7F]+', '', documents[i][:512])
    # Replace all spaces with underscores in id
    id = id.replace(" ", "_")

    # res = index.fetch(ids=[id], namespace="climate_claims_embeddings_unchanged")

    data.append(
            {
                "id": id,
                "values": embedding,
                "metadata": metadatas[i]
            }
    )
    if i % 100 == 0 and len(data) > 0:
        index.upsert(
            vectors=data,
            namespace= "climate_claims_embeddings_unchanged"
        )
        data = []
index.upsert(
    vectors=data,
    namespace= "climate_claims_embeddings_unchanged"
)