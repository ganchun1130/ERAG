# coding:utf-8
# @File  : retrieval_agent.py
# @Author: ganchun
# @Date  :  2025/04/23
# @Description: 基于通义千问的检索智能体

import os
from langchain_community.embeddings import ModelScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Dict, Any, Optional


class RetrievalAgent:
    def __init__(
            self,
            embedding_model: str = "damo/nlp_gte_sentence-embedding_chinese-base",
            top_k: int = 3
    ):
        """初始化检索智能体

        Args:
            embedding_model: 使用的嵌入模型名称
            top_k: 检索返回的文档数量
        """
        self.embedding = ModelScopeEmbeddings(model_name=embedding_model)
        self.vectorstore = None
        self.top_k = top_k
        self.text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

    def load_documents(self, documents: List[str], metadatas: Optional[List[Dict[str, Any]]] = None):
        """加载文档到知识库

        Args:
            documents: 文档内容列表
            metadatas: 文档元数据列表
        """
        docs = [Document(page_content=doc, metadata=meta or {})
                for doc, meta in zip(documents, metadatas or [{}] * len(documents))]
        splits = self.text_splitter.split_documents(docs)

        self.vectorstore = FAISS.from_documents(splits, self.embedding)
        print(f"已加载 {len(splits)} 个文档片段到知识库")

    def load_from_files(self, file_paths: List[str]):
        """从文件加载文档

        Args:
            file_paths: 文件路径列表
        """
        documents = []
        metadatas = []

        for file_path in file_paths:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append(content)
                    metadatas.append({"source": file_path})

        self.load_documents(documents, metadatas)

    def retrieve(self, query: str) -> List[Document]:
        """从知识库检索相关文档

        Args:
            query: 查询文本

        Returns:
            相关文档列表
        """
        if not self.vectorstore:
            raise ValueError("请先加载文档到知识库")

        docs = self.vectorstore.similarity_search(query, k=self.top_k)
        return docs

    def get_context(self, query: str) -> str:
        """获取检索上下文

        Args:
            query: 查询文本

        Returns:
            检索结果上下文文本
        """
        docs = self.retrieve(query)
        context = "\n\n".join([f"文档 {i + 1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
        return context