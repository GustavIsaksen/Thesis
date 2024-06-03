from typing import Optional
from docx import Document
from pydantic import BaseModel
from vertexai.language_models import TextEmbedding

class FetchedDocument(BaseModel):
    similarity: float # should be a number between 0 and 1
    response: str
    file_name: str
    metadata: Optional[dict] = None # contains keyvalue pairs like "Sheet name", "Topic", "Page number", etc.

    #TODO
    # def to_df(self) -> pd.Dataframe:
    #     return pd.DataFrame({
    #         "Similarity": [self.similarity],
    #         "Response": [self.response],
    #         "file name": [self.file_name],
    #         **self.metadata
    #     })

class Chunk(BaseModel):
    # TODO: Maybe this one needs metadata as well?
    text: str
    vector: Optional[list[TextEmbedding]] = None

    class Config:
        arbitrary_types_allowed = True

    def __repr__(self) -> str:
        return f"Chunk(page_content={self.chunk_content!r})"


class File(BaseModel):
    id: str
    file_name: str
    full_path: str
    file_type: str
    parent_folder : str
    content: Optional[Document] = None
    text_content: Optional[str] = None
    chunks: Optional[list[Chunk]] = []
    
    class Config:
        arbitrary_types_allowed = True

    def __repr__(self) -> str:
            content_type = type(self.content).__name__ if self.content else 'None'
            return f"File(id={self.id!r}, file_name={self.file_name!r}, full_path={self.full_path!r}, file_type={self.file_type!r}, parent_folder={self.parent_folder}, content={content_type})"

    def document_to_text(self) -> str:
        """Convert a Document object into a text string."""
        return "\n".join(paragraph.text for paragraph in self.content.paragraphs)