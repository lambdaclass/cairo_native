#!/usr/bin/env bash

# requires: curl, jq, ripgrep

latest_release=$(
    curl -L \
        -H "Accept: application/vnd.github+json" \
        -H "Authorization: Bearer $1"\
        -H "X-GitHub-Api-Version: 2022-11-28" \
        https://api.github.com/repos/starkware-libs/cairo/releases/latest \
        | jq -r .tag_name
  )

current_release=$(
    rg 'cairo-lang-sierra\s*=\s*(?:\{\s*git\s*=\s*"https://github.com/starkware-libs/cairo",\s*tag = "(.*?)"\s*\}|"(.*?)")' sierra2mlir/Cargo.toml -r '$1$2'
)

current_release_line=$(echo $current_release | cut -d':' -f1)
current_release_str=$(echo $current_release | cut -d':' -f2)

# Strip `v` prefix.
$current_release_str=$(echo $current_release_str | tr -d v)
$latest_release=$(echo $latest_release | tr -d v)

if [ "$current_release_str" != "$latest_release" && "v$current_release_str" != "$latest_release" ]; then
    echo "::warning file=sierra2mlir/Cargo.toml,line=$current_release_line,endLine=$current_release_line,title=Outdated cairo dependency::Current release = $current_release_str, Upstream release = $latest_release"
fi
