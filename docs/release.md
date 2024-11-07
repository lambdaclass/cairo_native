# Release Process

- Update version on root Cargo.toml
- Update version on runtime Cargo.toml
- Update Cargo.toml runtime dependency version to match the new version
- Make a PR
- Wait for PR to be merged
- Pull changes once merged
- `git tag v<version>`
- `git push --tags` # this will test, build and publish the version if successful.
