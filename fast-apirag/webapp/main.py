import os
import openai
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer
import pandas as pd
import torch
from dotenv import load_dotenv
load_dotenv()
cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())              # True
from openai import OpenAI
print(os.getenv("OPENAI_API_BASE"))
client = OpenAI(
        base_url= os.getenv("OPENAI_API_BASE") , # "http://<Your api-server IP>:port"
        api_key = os.getenv("OPENAI_API_KEY")
    )

url = "https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv"
df  = pd.read_csv(url)
df = df.sample(n=1000, random_state=42)  # Sample 1000 rows for faster processing
data = df.to_dict('records')  # Convert DataFrame to a list of dictionaries

encoder = SentenceTransformer('all-MiniLM-L6-v2', device=cuda)
# create the vector database client
qdrant = QdrantClient(":memory:") # Create in-memory Qdrant instance
# Create collection to store wines
qdrant.recreate_collection(
    collection_name="bbc_news",
    vectors_config=models.VectorParams(
        size=encoder.get_sentence_embedding_dimension(), # Vector size is defined by used model
        distance=models.Distance.COSINE
    )
)
# vectorize!
qdrant.upload_points(
    collection_name="bbc_news",
    points=[
        models.PointStruct(
            id=idx,
            vector=encoder.encode(doc["text"]).tolist(),
            payload=doc,
        ) for idx, doc in enumerate(data) # data is the variable holding all the bbc_news
    ]
)


app = FastAPI()


class Body(BaseModel):
    query: str


@app.get('/')
def root():
    return RedirectResponse(url='/docs', status_code=301)


@app.post('/ask')
def ask(body: Body):
    """
    Use the query parameter to interact with the Azure OpenAI Service
    using the Azure Cognitive Search API for Retrieval Augmented Generation.
    """
    search_result = search(body.query)
    chat_bot_response = assistant(body.query, search_result)
    return {'response': chat_bot_response}



def search(query):
    """
    Send the query to Azure Cognitive Search and return the top result
    """
    # Search time 

    hits = qdrant.search(
        collection_name="bbc_news",
        query_vector=encoder.encode(query).tolist(),
        limit=3
    )
    search_results = [hit.payload for hit in hits]
    search_results
    print(search_results)
    return search_results


def assistant(query, context):
   
    completion = client.chat.completions.create(
        model="LLaMA_CPP",
        messages=[
            {"role": "system", "content": "You are chatbot, a news specialist. Your top priority is to help guide users to the most relevant news articles based on their queries."},
            {"role": "user", "content": query},
            {"role": "assistant", "content": str(context)}
        ]
    )
    return completion.choices[0].message.content


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("webapp.main:app", host="127.0.0.1", port=8000, reload=True)
