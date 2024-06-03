from typing import Optional
from docx import Document
from pydantic import BaseModel
from langchain_core.vectorstores import VectorStore

class File(BaseModel):
    id: str
    file_name: str
    full_path: str
    file_type: str
    parent_folder : str
    content: Optional[Document] = None
    text_content: Optional[str] = None
    split_content: Optional[list[Document]] = None
    vector_store: Optional[VectorStore] = None

    class Config:
        arbitrary_types_allowed = True

    def __repr__(self) -> str:
            content_type = type(self.content).__name__ if self.content else 'None'
            return f"File(id={self.id!r}, file_name={self.file_name!r}, full_path={self.full_path!r}, file_type={self.file_type!r}, parent_folder={self.parent_folder}, content={content_type})"

    def document_to_text(self) -> str:
        """Convert a Document object into a text string."""
        return "\n".join(paragraph.text for paragraph in self.content.paragraphs)
    