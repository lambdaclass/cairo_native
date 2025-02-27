# Release Process

- Update version on root Cargo.toml
- Make a PR
- Wait for PR to be merged
- Pull changes once merged
- `git tag v<version>`
- `git push --tags` # this will test, build and publish the version if successful.
