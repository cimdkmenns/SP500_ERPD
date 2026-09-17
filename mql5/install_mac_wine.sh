#!/bin/bash
# Locate MetaTrader 5 data folders inside macOS Wine bottles and install the
# Elder-Ray scalper EA and its presets into them.
#
#   ./install_mac_wine.sh              find folders, show a plan, ask, install
#   ./install_mac_wine.sh --dry-run    find and show the plan only
#   ./install_mac_wine.sh --yes        install without asking
#   ./install_mac_wine.sh --root DIR   also search DIR for bottles
#
# Written for the bash 3.2 that ships with macOS: no associative arrays,
# no mapfile. Existing files are backed up, never silently overwritten.

set -u

SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
EA="ElderRay_Scalper_M2_M5_EA.mq5"
PRESETS="ElderRay_Scalper_baseline.set ElderRay_Scalper_sweep.set"

DRY_RUN=0
ASSUME_YES=0
EXTRA_ROOT=""

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --yes|-y)  ASSUME_YES=1 ;;
    --root)    shift; EXTRA_ROOT="${1:-}" ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

if [ ! -f "$SRC_DIR/$EA" ]; then
  echo "error: $EA not found next to this script (looked in $SRC_DIR)." >&2
  exit 1
fi

# Wine prefix roots used by the common ways of running MT5 on a Mac.
ROOTS="
$HOME/Library/Application Support/net.metaquotes.wine.metatrader5
$HOME/Library/Application Support/MetaTrader 5
$HOME/Library/Application Support/CrossOver/Bottles
$HOME/Library/PlayOnMac/wineprefix
$HOME/Library/Containers/com.isaacmarovitz.Whisky/Bottles
$HOME/Applications
$HOME/.wine
"
[ -n "$EXTRA_ROOT" ] && ROOTS="$ROOTS
$EXTRA_ROOT"

echo "Searching for MT5 data folders (this can take a few seconds)..."
FOUND_LIST="$(mktemp)"
trap 'rm -f "$FOUND_LIST"' EXIT

echo "$ROOTS" | while IFS= read -r root; do
  [ -z "$root" ] && continue
  [ -d "$root" ] || continue
  # An MT5 data folder is the parent of an MQL5 directory containing Experts.
  find "$root" -maxdepth 16 -type d -name Experts -path '*/MQL5/Experts' \
       -print 2>/dev/null
done | sed 's|/MQL5/Experts$||' | sort -u > "$FOUND_LIST"

COUNT=$(wc -l < "$FOUND_LIST" | tr -d ' ')
if [ "$COUNT" = "0" ]; then
  cat >&2 <<'MSG'

No MT5 data folder found in the usual Wine locations.

Get the exact path from MT5 itself: attach the EA (or run a backtest) and
read the first Journal line. It prints, for example:

  MT5 data folder (...): C:\users\you\AppData\Roaming\MetaQuotes\Terminal\<hash>

Map that Windows path onto the bottle, then re-run with:

  ./install_mac_wine.sh --root "/path/to/the/bottle"

MSG
  exit 1
fi

echo "Found $COUNT:"
i=0
while IFS= read -r dir; do
  i=$((i+1))
  echo "  [$i] $dir"
done < "$FOUND_LIST"

echo
echo "Plan for each folder above:"
echo "  copy $EA -> MQL5/Experts/"
for f in $PRESETS; do
  [ -f "$SRC_DIR/$f" ] && echo "  copy $f -> MQL5/Presets/"
done
echo "  any existing file of the same name is renamed to *.bak-<timestamp>"

if [ "$DRY_RUN" = "1" ]; then
  echo
  echo "Dry run - nothing written."
  exit 0
fi

if [ "$ASSUME_YES" != "1" ]; then
  echo
  printf "Install into all %s folder(s)? [y/N] " "$COUNT"
  read -r reply
  case "$reply" in
    y|Y|yes|YES) ;;
    *) echo "Cancelled."; exit 0 ;;
  esac
fi

STAMP="$(date +%Y%m%d-%H%M%S)"

install_one() {
  src="$1"; dest_dir="$2"
  base="$(basename "$src")"
  mkdir -p "$dest_dir" || return 1
  if [ -f "$dest_dir/$base" ]; then
    mv "$dest_dir/$base" "$dest_dir/$base.bak-$STAMP" || return 1
    echo "      backed up existing $base -> $base.bak-$STAMP"
  fi
  cp "$src" "$dest_dir/$base" || return 1
  echo "      installed $base"
}

FAILED=0
while IFS= read -r dir; do
  echo "  $dir"
  install_one "$SRC_DIR/$EA" "$dir/MQL5/Experts" || FAILED=1
  for f in $PRESETS; do
    [ -f "$SRC_DIR/$f" ] || continue
    install_one "$SRC_DIR/$f" "$dir/MQL5/Presets" || FAILED=1
  done
done < "$FOUND_LIST"

echo
if [ "$FAILED" = "1" ]; then
  echo "Finished with errors - check the messages above." >&2
  exit 1
fi
cat <<'MSG'
Done. In MT5:
  1. MetaEditor -> open ElderRay_Scalper_M2_M5_EA.mq5 -> Compile (F7)
  2. Strategy Tester -> Inputs -> Load -> ElderRay_Scalper_baseline.set
  3. Symbol NAS100, period M5, model "Every tick based on real ticks"
MSG
