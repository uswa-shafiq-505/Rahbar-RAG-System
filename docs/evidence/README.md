# Evidence

Commit the real output here (`make evidence` captures most of it):

- `kubectl-top.txt`       `kubectl top pods -n rahbar --containers` while the indexer runs and while idle
- `endpoints.txt`         `kubectl get endpoints api -n rahbar` after scaling to 3 or more
- `hpa.txt`               `kubectl get hpa -n rahbar` showing a real percentage, not `<unknown>`
- `branch-protection.png` GitHub settings for main
- `blocked-merge.png`     a pull request with a failing eval gate and the greyed-out merge button
- `green-merge.png`       the same pull request after the fix
- `pin-mismatch.txt`      `kubectl get pods` at 0/1 plus the log line, from the section 4.4 demo
- `rollout-undo.txt`      `kubectl rollout status` and `undo`
