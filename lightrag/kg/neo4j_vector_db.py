import asyncio
import os
from typing import Any, final, List
from dataclasses import dataclass
import hashlib
import uuid
from neo4j import GraphDatabase
import configparser
import pipmaster as pm
from ..utils import logger, compute_mdhash_id
from ..base import BaseVectorStorage
import numpy as np

# Ensure the required Neo4j client is installed
if not pm.is_installed("neo4j"):
    pm.install("neo4j")

# Read configuration
config = configparser.ConfigParser()
config.read("config.ini", "utf-8")


@final
@dataclass
class Neo4jVectorDBStorage(BaseVectorStorage):
    def __post_init__(self):
        kwargs = self.global_config.get("vector_db_storage_cls_kwargs", {})
        cosine_threshold = kwargs.get("cosine_better_than_threshold")
        if cosine_threshold is None:
            raise ValueError(
                "cosine_better_than_threshold must be specified in vector_db_storage_cls_kwargs"
            )
        self.cosine_better_than_threshold = cosine_threshold
        self._neo4j_database = os.environ.get(
            "NEO4J_DATABASE",
            config.get("Chunk-entity-relation", "database", fallback=None),
        )

        # Initialize Neo4j client
        self._client = GraphDatabase.driver(
            os.environ.get("NEO4J_URI", config.get("neo4j", "uri", fallback=None)),
            auth=(
                os.environ.get(
                    "NEO4J_USERNAME", config.get("neo4j", "user", fallback=None)
                ),
                os.environ.get(
                    "NEO4J_PASSWORD", config.get("neo4j", "password", fallback=None)
                ),
            ),
        )

        self._max_batch_size = self.global_config["embedding_batch_num"]

    def _create_node_query(self, id: str, vector: list[float], payload: dict) -> str:
        """Create a Cypher query to insert a vector node into Neo4j."""
        query = """
        MERGE (n:Vector {id: $id})
        SET n.vector = $vector, n.payload = $payload
        """
        query_match_entity = """
        MATCH (e:base {entity_id: $entity_name})
        MERGE (v:Vector {id: $id})
        SET v.vector = $vector, v.payload = $payload
        MERGE (v)-[:EMBEDDING_OF]->(e)
        """
        import json

        payload_json = json.dumps(payload)

        import json

        payload_json = json.dumps(payload)

        if "entity_name" in payload:
            return query_match_entity, {
                "id": id,
                "vector": vector,
                "payload": payload_json,
                "entity_name": payload["entity_name"],
            }

        params = {"id": id, "vector": vector, "payload": payload_json}
        return query, params

    @staticmethod
    def create_collection_if_not_exist(
        client: GraphDatabase, collection_name: str, **kwargs
    ):
        """Ensure the collection (in this case, a node type) exists."""
        # Neo4j doesn't require explicit collection creation; we rely on MERGE for node existence
        pass

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        logger.info(f"Inserting {len(data)} into Neo4j")
        if not data:
            return

        list_data = [
            {
                "id": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]

        embedding_tasks = [self.embedding_func(batch) for batch in batches]
        embeddings_list = await asyncio.gather(*embedding_tasks)

        embeddings = np.concatenate(embeddings_list)
        # Perform batch insertions into Neo4j using run_in_executor to handle synchronous session calls
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._insert_into_neo4j, list_data, embeddings)

        logger.info(f"Successfully upserted {len(list_data)} vectors into Neo4j")

    def _insert_into_neo4j(self, list_data, embeddings):
        with self._client.session(database=self._neo4j_database) as session:
            for i, d in enumerate(list_data):
                node_id = compute_mdhash_id(d["id"])
                query, params = self._create_node_query(
                    node_id, embeddings[i].tolist(), d
                )
                session.run(query, **params)

    async def query(
        self, query: str, top_k: int, ids: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Query the database for the top K similar vectors."""

        embedding = await self.embedding_func([query])

        # Retrieve the top K most similar vectors from Neo4j
        with self._client.session(database=self._neo4j_database) as session:
            cypher_query = """
            MATCH (n:Vector)
            WHERE n.vector IS NOT NULL
            WITH n, gds.similarity.cosine(n.vector, $embedding) AS similarity
            ORDER BY similarity DESC
            LIMIT $top_k
            RETURN n.id AS id, n.payload AS payload, similarity
            """
            # Execute the query with parameters
            results = session.run(
                cypher_query, embedding=embedding[0].tolist(), top_k=top_k
            )

            # Parse results
            result_list = []
            import json

            for record in results:
                # Parse the payload from JSON string
                payload = json.loads(record["payload"])

                # Extract the keys from the payload
                record_data = {
                    "id": payload.get("id"),
                    "entity_name": payload.get("entity_name"),
                    "content": payload.get("content"),
                    "source_id": payload.get("source_id"),
                    "file_path": payload.get("file_path"),
                    "full_doc_id": payload.get("full_doc_id"),
                    "src_id": payload.get("src_id"),
                    "tgt_id": payload.get("tgt_id"),
                }

                # Include the similarity score in the record data
                record_data["similarity"] = record["similarity"]

                # Append the processed record to the result list
                if record_data["entity_name"]:
                    result_list.append(record_data)

            return result_list

    async def delete_entity(self, entity_name: str) -> None:
        """Delete an entity by name."""
        try:
            entity_id = compute_mdhash_id(entity_name, prefix="ent-")
            logger.debug(
                f"Attempting to delete entity {entity_name} with ID {entity_id}"
            )
            with self._client.session(database=self._neo4j_database) as session:
                cypher_query = f"""
                MATCH (n:Vector {{id: '{entity_id}'}})
                DELETE n
                """
                session.run(cypher_query)
            logger.debug(f"Successfully deleted entity {entity_name}")
        except Exception as e:
            logger.error(f"Error deleting entity {entity_name}: {e}")

    async def delete_entity_relation(self, entity_name: str) -> None:
        """Delete all relations associated with an entity."""
        try:
            with self._client.session(database=self._neo4j_database) as session:
                cypher_query = f"""
                MATCH (src:Vector)-[r]->(tgt:Vector)
                WHERE src.id = '{entity_name}' OR tgt.id = '{entity_name}'
                DELETE r
                """
                session.run(cypher_query)
            logger.debug(f"Successfully deleted relations for entity {entity_name}")
        except Exception as e:
            logger.error(f"Error deleting relations for entity {entity_name}: {e}")

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get multiple vectors by their IDs."""
        if not ids:
            return []

        try:
            with self._client.session(database=self._neo4j_database) as session:
                cypher_query = f"""
                MATCH (n:Vector)
                WHERE n.id IN {ids}
                RETURN n.payload AS payload, n.vector AS vector
                """
                results = session.run(cypher_query)

            return [
                {"payload": record["payload"], "vector": record["vector"]}
                for record in results
            ]
        except Exception as e:
            logger.error(f"Error retrieving vector data for IDs {ids}: {e}")
            return []

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get vector data by its ID."""
        try:
            with self._client.session(database=self._neo4j_database) as session:
                cypher_query = f"""
                MATCH (n:Vector)
                WHERE n.id = '{id}'
                RETURN n.payload AS payload, n.vector AS vector
                """
                result = session.run(cypher_query)

            # If a result is found
            if result.peek():
                record = result.single()
                return {"payload": record["payload"], "vector": record["vector"]}
            else:
                return None

        except Exception as e:
            logger.error(f"Error retrieving vector data for ID {id}: {e}")
            return None

    async def delete(self, ids: List[str]) -> None:
        """Delete vectors with specified IDs."""
        try:
            with self._client.session(database=self._neo4j_database) as session:
                cypher_query = f"""
                MATCH (n:Vector)
                WHERE n.id IN {ids}
                DELETE n
                """
                session.run(cypher_query)
            logger.debug(f"Successfully deleted {len(ids)} vectors from Neo4j")
        except Exception as e:
            logger.error(f"Error while deleting vectors: {e}")

    async def drop(self) -> dict[str, str]:
        """Drop all vector data from storage and clean up resources."""
        try:
            # Drop all vectors from the collection
            with self._client.session(database=self._neo4j_database) as session:
                cypher_query = "MATCH (n:Vector) DELETE n"
                session.run(cypher_query)

            logger.info(f"All data dropped from Neo4j collection {self.namespace}")
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(
                f"Error dropping data from Neo4j collection {self.namespace}: {e}"
            )
            return {"status": "error", "message": str(e)}

    async def index_done_callback(self) -> None:
        """Placeholder for any callback actions after index completion"""
        pass
