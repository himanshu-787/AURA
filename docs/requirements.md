# AURA Requirements

## Goal
Answer research questions using evidence, with citations.

## v0.1 Must do
- Load text/PDF documents from the data/ folder
- Split them into chunks and store embeddings
- Given a question, find the most relevant chunks
- Generate an answer that cites which chunk it used

## v0.1 Will NOT do
- No web search, no agents, no login, no frontend
- No GPU training

## Success check
For 10 test questions, the answer must be supported by the cited text.