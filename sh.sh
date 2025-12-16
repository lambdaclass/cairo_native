#!/usr/bin/env bash
# Retrieve last release tag and process it to make it usable
          last_release=$(curl -s https://api.github.com/repos/starkware-libs/cairo/releases/latest \
              | jq -r .tag_name \
              | cut -c 2-
          )

          # Updates cairo-lang dependency to the last release
          name='cairo-lang-[[:alnum:]-]+'
          version='"([[:alnum:].~-])+"'
          
          sed -i '' -r "s/^($name) = $version/\1 = \"~$last_release\"/" Cargo.toml
