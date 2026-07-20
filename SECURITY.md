# Security operations

## Exposed football-data credential

The football-data credential previously committed to this repository must be treated
as compromised. The current source tree no longer uses or contains that value, but
removing it from the latest revision does not invalidate copies in Git history,
forks, caches, or previously built deployment artifacts.

Before the backend is redeployed, the account owner must:

1. Revoke the exposed credential through the football-data account or provider support.
2. Generate a replacement that has never appeared in source control.
3. Store the replacement in Google Secret Manager.
4. Grant only the Cloud Run service identity access to the secret.
5. Bind a pinned secret version to the `FOOTBALL_API_TOKEN` environment variable.
6. Deploy a new revision and confirm API-backed routes work.
7. Confirm old Cloud Run revisions and cached artifacts cannot use the revoked value.

Do not reactivate or reuse the old credential. This repository change cannot perform
provider-account rotation and does not claim that revocation has occurred.

## Git history

A local exact-value audit found the exposed value in 45 reachable commits without
printing it. History has not been rewritten: doing so changes commit identities and
requires collaborator coordination plus an explicitly approved force-push. Revocation
is the immediate containment step even if a later coordinated history purge is chosen.

## Local and production configuration

- Export `FOOTBALL_API_TOKEN` in the local process environment when provider access is needed.
- Keep local `.env` variants ignored; `.env.example` intentionally contains no value.
- Do not pass the replacement as a Docker build argument or bake it into an image layer.
- Do not place it in `app.yaml`, workflow files, logs, screenshots, tests, or fixtures.
- In Cloud Run, reference a pinned Secret Manager version as the environment variable.
- Rotate again immediately if the replacement is ever exposed.
