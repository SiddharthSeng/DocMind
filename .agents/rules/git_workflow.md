# Git Workflow Rules

After completing each build step (creating or modifying files), automatically follow this git workflow:

1. Double-check for unintended files: Use `git status` to verify no large files, secrets (`.env`), or gitignored files are being staged. If anything looks incorrect, stop and inform the user instead of proceeding.
2. Stage all changes: `git add .`
3. Commit the changes: `git commit -m "<message>"`
   - The message must clearly and specifically describe the functionality added (e.g., "Add sentence-transformers embedding pipeline and ChromaDB vector store").
   - **CRITICAL**: Do NOT include step numbers (like "step 3" or "Step 1-2") in the commit message.
4. Push to remote: `git push`

Always execute this automatically for every build step unless the user explicitly asks to pause it.
