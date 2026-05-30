# RAG Source Documents

Place your system-approved knowledge files in this folder.

Supported formats:
- PDF
- TXT
- JSON

Recommended files:
- nutrition_basics.txt
- meal_planning.txt
- fat_loss.txt
- muscle_gain.txt
- hydration.txt
- ACSM / ISSN / NSCA guideline PDFs

How indexing works:
1. Files in this folder are loaded by the RAG pipeline.
2. Run POST /rag/reindex to rebuild FAISS after updates.
3. The vector index is stored under knowledge_base/faiss_index.
