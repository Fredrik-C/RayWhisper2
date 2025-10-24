I want to build a voice to text tool
it should use state-of-the-art model for fast and precise voice to text
it should have a start listening and stop listening based on keyboard shortcuts
it should use a vector database with embeddings to guide the output
there should be a tool to populate the embedding database by scanning a folder of markdown files and .cs files.
the RAG retrieval from the embedding database should also have a reranking
this voice to Text tool should work on Windows machines as well as Mac OS, it is important that the toggling from the keyboard shortcuts works very well in both Windows and macOS
the text output should preferably be keystrokes so we do not lose any content in the clipboard, so avoid using clipboard for the output it should be keyboard.