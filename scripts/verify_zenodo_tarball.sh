#!/bin/bash
# Sanity-check the Zenodo tarball before uploading.

set -e
WORK=${WORK:?Set WORK to the directory containing the archive}
TARBALL=${1:-$WORK/zenodo_metatune_v1.tar.gz}
VERIFY_DIR=$WORK/zenodo_verify

[ -f "$TARBALL" ] || { echo "ERROR: $TARBALL not found"; exit 1; }
rm -rf "$VERIFY_DIR"; mkdir -p "$VERIFY_DIR"

echo "[1] Tarball integrity"
tar -tzf "$TARBALL" > /dev/null && echo "    ok"
echo "    size: $(du -h "$TARBALL" | cut -f1)"

echo "[2] Extracting..."
tar -xzf "$TARBALL" -C "$VERIFY_DIR"
ROOT=$(ls "$VERIFY_DIR")
echo "    root: $ROOT"

echo "[3] Layout check"
for d in semantic_main ablations/vanilla_joint ablations/swap_meta; do
    n=$(ls "$VERIFY_DIR/$ROOT/$d" 2>/dev/null | wc -l)
    echo "    $d/  ($n run dirs)"
done
[ -f "$VERIFY_DIR/$ROOT/README_zenodo.md" ] && echo "    README_zenodo.md present"

echo "[4] Per-run integrity (every run dir must have best.pth + config.txt)"
n_missing=0
for d in $(find "$VERIFY_DIR/$ROOT" -mindepth 2 -type d -name "*_img256_*"); do
    [ -f "$d/best.pth" ]  || { echo "    MISSING best.pth: $(basename $d)";  n_missing=$((n_missing + 1)); }
    [ -f "$d/config.txt" ] || { echo "    MISSING config.txt: $(basename $d)"; n_missing=$((n_missing + 1)); }
done
[ "$n_missing" = 0 ] && echo "    ok ($(find $VERIFY_DIR/$ROOT -name best.pth | wc -l) checkpoints, all with config.txt)"

echo "[5] Allow-list check — every run-dir basename should match one of these prefixes:"
echo "    blood4 | osteosarcoma4 | cellBT4744 | cellHuh74 | multimodal4 | cyto4 |"
echo "    yeast-bright4 | yeast-contrast4 | fluocellRed10 | sartorius10"
allow_re='^(blood4|osteosarcoma4|cellBT4744|cellHuh74|multimodal4|cyto4|yeast-bright4|yeast-contrast4|fluocellRed10|sartorius10)_'
n_bad=0
for d in $(find "$VERIFY_DIR/$ROOT" -mindepth 2 -type d -name "*_img256_*"); do
    bn=$(basename "$d")
    if ! echo "$bn" | grep -qE "$allow_re"; then
        echo "    NOT IN ALLOWLIST: $bn"
        n_bad=$((n_bad + 1))
    fi
done
[ "$n_bad" = 0 ] && echo "    ok (every run matches the paper's (dataset,N) configurations)"

echo "[6] Torch loadability check on a random checkpoint"
random_ckpt=$(find "$VERIFY_DIR/$ROOT" -name "best.pth" | shuf | head -1)
echo "    trying: $(echo $random_ckpt | sed "s|$VERIFY_DIR/$ROOT/||")"
${PYTHON:-python} -c "
import torch
sd = torch.load('$random_ckpt', map_location='cpu', weights_only=False)
print(f'    state_dict has {len(sd)} keys; first 5:', list(sd.keys())[:5])
print('    loadable: ok')
" 2>&1 | tail -3

echo
echo "All checks passed. Tarball ready to upload."
echo "  file: $TARBALL"
echo "  size: $(du -h "$TARBALL" | cut -f1)"
echo
echo "Clean up the verification dir with:"
echo "  rm -rf $VERIFY_DIR"
