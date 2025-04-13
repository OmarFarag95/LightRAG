import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import (
    gpt_4o_mini_complete,
    gpt_4o_complete,
    openai_embed,
    openai_complete_if_cache,
)
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger
from lightrag.utils import EmbeddingFunc
from dotenv import load_dotenv

setup_logger("lightrag", level="INFO")
setup_logger("lightrag", level="DEBUG")
load_dotenv()
WORKING_DIR = r"AI/src/graphrag"


api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("BASE_URL")

#NOTE: importan to run on windows
import nest_asyncio
nest_asyncio.apply()


async def embedding_func(texts):
    return await openai_embed(
        texts=texts,
        model="intfloat/multilingual-e5-large",
        base_url=base_url,
        api_key=api_key,
    )


async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    return await openai_complete_if_cache(
        model="Qwen/Qwen2.5-72B-Instruct",
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        base_url=base_url,
        api_key=api_key,
        **kwargs
    )


async def initialize_rag():
    # Initialize LightRAG with Neo4j as the graph storage
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        graph_storage="Neo4JStorage",
        log_level="INFO",
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,  # For 'text-embedding-ada-002'
            max_token_size=8192,  # Adjust based on your requirements
            func=embedding_func,
        ),
        vector_storage="Neo4jVectorDBStorage",
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag


def main():
    rag = asyncio.run(initialize_rag())

    # load sample_data.json
    import json
    with open(r"C:\Users\omarhamed\Desktop\test_workspace\AI\src\graphrag\sample_data.json", "r", encoding="utf-8") as f:
        sample_data = json.load(f)
    
    for data in sample_data[:1]:
        query = data["question"]
        # join list with comma
        answer = data["answers"][:1]
        answer = '\\n'.join(answer)
        rag.insert(query)
        rag.insert(answer)
        exit()
        


    


if __name__ == "__main__":
    main()
