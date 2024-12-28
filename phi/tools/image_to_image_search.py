from phi.tools.openai_embeddings import OpenAIEmbeddings
from phi.vectordb.chroma import ChromaDb
from phi.document import Document
from typing import List
import numpy as np

class ImageSearcher:
    def __init__(self):
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings()
        
        # Initialize ChromaDB
        self.vector_db = ChromaDb(
            collection="image_collection",
            embedder=self.embeddings,
            path="tmp/image_db",
            persistent_client=True
        )
        
        # Create the collection
        self.vector_db.create()
    
    def add_images(self, image_urls: List[str], clear_existing: bool = False):
        """Add images to the vector database"""
        if clear_existing:
            self.vector_db.drop()
            self.vector_db.create()
        
        # Create Document objects for each image
        image_docs = []
        for i, url in enumerate(image_urls):
            doc = Document(
                content=url,
                name=f"image_{i}",
                meta_data={"type": url.split('/')[-1].split('_')[0]}  # Extract image type from URL
            )
            image_docs.append(doc)
        
        # Insert documents into the vector database
        self.vector_db.insert(image_docs)
        print(f"Added {len(image_docs)} images to the database")
    
    def search_similar_images(self, query_image_url: str, category: str = None, limit: int = 5) -> List[dict]:
        """Search for similar images given a query image URL and optional category"""
        # Get embedding for query image
        self.embeddings.load_model()
        query_embedding = self.embeddings.get_image_embeddings(query_image_url)
        
        # Get raw collection to access embeddings
        chroma_collection = self.vector_db._collection
        collection_data = chroma_collection.get(
            include=['embeddings', 'documents', 'metadatas']
        )
        
        # Calculate similarities manually
        similar_images = []
        for doc_content, doc_embedding in zip(collection_data['documents'], collection_data['embeddings']):
            if doc_content != query_image_url:  # Skip the query image
                # Calculate cosine similarity
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                print("similarity", similarity)
                distance = 1 - similarity
                
                similar_images.append({
                    'url': doc_content,
                    'distance': distance,
                    'type': doc_content.split('/')[-1].split('_')[0]
                })
        
        # Sort by distance
        similar_images.sort(key=lambda x: x['distance'])
        return similar_images[:limit]
