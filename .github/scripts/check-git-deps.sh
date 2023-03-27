#!/usr/bin/env bash

# requires: curl, jq, ripgrep

set -x

latest_release=$(
    curl -L \
        -H "Accept: application/vnd.github+json" \
        -H "Authorization: Bearer $GITHUB_TOKEN"\
        -H "X-GitHub-Api-Version: 2022-11-28" \
        https://api.github.com/repos/starkware-libs/cairo/releases/latest \
        | jq .tag_name
  )

current_release=$(
    rg -N 'cairo-lang-sierra = \{ git = "https://github.com/starkware-libs/cairo", branch = "(.*?)" \}' sierra2mlir/Cargo.toml -r '$1'
)

if [ "$current_release" != "$latest_release" ]; then
    echo "::warning file=sierra2mlir/Cargo.toml,line=16,endLine=16,title=Outdated cairo dependency::Current release = $current_release, Upstream release = $latest_release"
fi
