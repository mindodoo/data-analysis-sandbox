#!/usr/bin/env bash
# Set up Kaggle credentials for the CLI (kaggle 1.7.x uses legacy username + key).
#
# IMPORTANT: "Create New Token" on Kaggle gives a KGAT_... token — that does NOT
# download kaggle.json. For this CLI you need a LEGACY API key instead:
#   https://www.kaggle.com/settings -> "Legacy API Credentials" -> "Create Legacy API Key"
# That downloads kaggle.json to ~/Downloads/

set -euo pipefail

KAGGLE_DIR="$HOME/.kaggle"
CRED="$KAGGLE_DIR/kaggle.json"

mkdir -p "$KAGGLE_DIR"
chmod 700 "$KAGGLE_DIR"

install_json() {
  local src="$1"
  cp "$src" "$CRED"
  chmod 600 "$CRED"
  echo "Installed: $CRED"
  echo "Test with: kaggle kernels list --page-size 1"
}

if [[ -f "$CRED" ]]; then
  chmod 600 "$CRED"
  echo "Credentials already exist: $CRED"
  exit 0
fi

# 1) Legacy key downloaded from Kaggle
for candidate in \
  "$HOME/Downloads/kaggle.json" \
  "$HOME/Desktop/kaggle.json" \
  "$(dirname "$0")/../kaggle.json"; do
  if [[ -f "$candidate" ]]; then
    install_json "$candidate"
    exit 0
  fi
done

# 2) Environment variables (legacy format)
if [[ -n "${KAGGLE_USERNAME:-}" && -n "${KAGGLE_KEY:-}" ]]; then
  printf '{"username":"%s","key":"%s"}\n' "$KAGGLE_USERNAME" "$KAGGLE_KEY" > "$CRED"
  chmod 600 "$CRED"
  echo "Created $CRED from KAGGLE_USERNAME + KAGGLE_KEY"
  exit 0
fi

# 3) Interactive entry
if [[ -t 0 ]]; then
  echo ""
  echo "Kaggle changed their API. The main 'Create New Token' button gives a KGAT_ token"
  echo "that does NOT work with this CLI. You need a LEGACY key instead:"
  echo ""
  echo "  1. Open https://www.kaggle.com/settings"
  echo "  2. Scroll to 'Legacy API Credentials'"
  echo "  3. Click 'Create Legacy API Key' (downloads kaggle.json)"
  echo "  4. Re-run this script"
  echo ""
  echo "Or enter your legacy credentials now (from kaggle.json):"
  echo ""
  read -r -p "Kaggle username: " username
  read -r -s -p "Legacy API key (long hex string, NOT KGAT_...): " apikey
  echo ""
  if [[ -n "$username" && -n "$apikey" ]]; then
    if [[ "$apikey" == KGAT_* ]]; then
      echo "Error: That looks like a new KGAT token. Use 'Create Legacy API Key' on Kaggle settings."
      exit 1
    fi
    printf '{"username":"%s","key":"%s"}\n' "$username" "$apikey" > "$CRED"
    chmod 600 "$CRED"
    echo "Created: $CRED"
    exit 0
  fi
fi

cat <<'EOF'

No kaggle.json found.

Kaggle's "Create New Token" (KGAT_...) does not download a file.
Use the LEGACY key instead:

  https://www.kaggle.com/settings
  -> Legacy API Credentials
  -> Create Legacy API Key

That saves kaggle.json to Downloads. Then run:

  /Users/mindodoo/Projects/data_ana/scripts/setup-kaggle-credentials.sh

Or create the file manually:

  mkdir -p ~/.kaggle && chmod 700 ~/.kaggle
  nano ~/.kaggle/kaggle.json

  {
    "username": "YOUR_KAGGLE_USERNAME",
    "key": "YOUR_LEGACY_API_KEY"
  }

  chmod 600 ~/.kaggle/kaggle.json

EOF
exit 1
