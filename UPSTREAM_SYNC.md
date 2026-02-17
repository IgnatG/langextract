# Upstream Sync Maintenance

Recurring task to keep the managed fork (`ignatg/langextract`) current with Google's upstream `google/langextract`.

## Schedule

Run **monthly**, or whenever:

- Google releases a new version of langextract
- An upstream PR you cherry-picked gets merged (your cherry-pick becomes a no-op → diff shrinks)
- You need a new upstream feature or bugfix

## Prerequisites

```bash
# Ensure both remotes are configured (one-time)
cd langextract/
git remote -v
# origin    https://github.com/ignatg/langextract.git (fetch/push)
# upstream  https://github.com/google/langextract.git (fetch)

# If upstream is missing:
git remote add upstream https://github.com/google/langextract.git
```

## Sync Procedure

### 1. Sync `main` from upstream

```bash
git checkout main
git pull upstream main
git push origin main
```

### 2. Rebase `custom` onto updated `main`

```bash
git checkout custom
git rebase main
```

During rebase:

- Cherry-picks for **merged PRs** auto-resolve (empty commits are skipped) → diff shrinks naturally
- Cherry-picks for **unmerged PRs** may have minor conflicts → resolve manually
- If a conflict is trivial, resolve and `git rebase --continue`
- If a conflict is complex, `git rebase --abort` and investigate

### 3. Force-push `custom`

```bash
git push origin custom --force-with-lease
```

### 4. Update downstream lockfile

```bash
cd ../document-queue
pip install --upgrade -e ../langextract
# Or if using git+https in production:
pip install --upgrade "langextract @ git+https://github.com/ignatg/langextract.git@custom"
```

### 5. Run tests

```bash
cd ../document-queue
python -m pytest tests/
```

## Removing merged cherry-picks

When a PR you cherry-picked is merged upstream:

1. After rebasing `custom` onto updated `main`, the cherry-pick commit will be empty
2. Git may auto-skip it, or you can `git rebase --skip`
3. If all cherry-picked PRs are merged → `custom` branch becomes identical to `main`
4. At that point, switch `document-queue` back to upstream: `pip install langextract` (from PyPI)

## Troubleshooting

| Issue | Solution |
| ----- | -------- |
| Rebase conflict | Check if the upstream change overlaps with your cherry-pick. If the PR was merged differently, drop your cherry-pick (`git rebase --skip`) |
| Tests fail after sync | Pin to a specific upstream commit: `git rebase <safe-commit>` instead of `main` |
| Need a new PR | `git fetch upstream pull/<PR#>/head:pr-<PR#>` then `git cherry-pick <commits>` onto `custom` |
