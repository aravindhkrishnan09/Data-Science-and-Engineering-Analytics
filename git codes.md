Step 1: Check the last commit (details)

Use this to see what was committed, by whom, and when.

git log -1


If you want a concise, readable view:

git log -1 --oneline


If you want to see file-level changes in that commit:

git show

Step 2: Decide rollback strategy

Be explicit about intent before acting.

A. Roll back last commit, keep changes staged
git reset --soft HEAD~1


Use when the commit message or grouping was wrong.

B. Roll back last commit, keep changes unstaged (most common)
git reset HEAD~1


Use when you need to rework the changes.

C. Roll back last commit and discard changes
git reset --hard HEAD~1


Use only when changes are garbage and unrecoverable.

D. Commit already pushed to remote (safe rollback)
git revert HEAD


Creates a clean inverse commit—no history rewrite.

Step 3: Verify state after rollback

Always confirm.

git status
git log -1 --oneline