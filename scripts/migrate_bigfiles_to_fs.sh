#!/bin/bash
# Move large, finished artefacts from the data disk to /root/autodl-fs and leave symlinks behind.
# Per item: rsync copy -> verify (file count, symlink count, apparent bytes) -> delete local -> symlink.
# On any mismatch the local copy is kept untouched and the item is marked FAIL.
set -u
P=/root/autodl-tmp/Uveitis-Segmentation-bs
DST=/root/autodl-fs/uveitis_bigfiles_20261015
MAN=$DST/MIGRATION_MANIFEST.tsv
mkdir -p $DST/outputs $DST/runs $DST/misc $DST/outputs/experiment_archives
[ -f $MAN ] || printf "time\tstatus\tsource\tdest\tfiles\tlinks\tbytes\n" > $MAN
log(){ echo "[$(date +%F' '%T)] $*"; }
stat3(){ echo "$(find "$1" -type f | wc -l) $(find "$1" -type l | wc -l) $(find "$1" -type f -printf '%s\n' | awk '{s+=$1} END{printf "%.0f", s}')"; }

move_dir(){  # $1 = source dir, $2 = dest dir
  local s=$1 d=$2
  if [ -L "$s" ]; then log "SKIP (already symlink) $s"; return; fi
  [ -d "$s" ] || { log "SKIP (missing) $s"; return; }
  log "COPY $s -> $d"
  rsync -a "$s/" "$d/" || { log "FAIL rsync $s"; printf "%s\tFAIL_RSYNC\t%s\t%s\t-\t-\t-\n" "$(date +%FT%T)" "$s" "$d" >> $MAN; return; }
  local a b; a=$(stat3 "$s"); b=$(stat3 "$d")
  if [ "$a" != "$b" ]; then
    log "FAIL verify $s src=[$a] dst=[$b]"; printf "%s\tFAIL_VERIFY\t%s\t%s\t%s\n" "$(date +%FT%T)" "$s" "$d" "$a | $b" >> $MAN; return
  fi
  rm -rf "$s" && ln -s "$d" "$s"
  set -- $a; printf "%s\tOK\t%s\t%s\t%s\t%s\t%s\n" "$(date +%FT%T)" "$s" "$d" "$1" "$2" "$3" >> $MAN
  log "OK $s files=$1 links=$2 bytes=$3"
}

move_file(){  # $1 = source file, $2 = dest file
  local s=$1 d=$2
  if [ -L "$s" ] || [ ! -f "$s" ]; then log "SKIP $s"; return; fi
  log "COPY $s"
  cp -p "$s" "$d" || { log "FAIL cp $s"; return; }
  if [ "$(md5sum < "$s")" != "$(md5sum < "$d")" ]; then log "FAIL md5 $s"; printf "%s\tFAIL_MD5\t%s\t%s\t1\t0\t-\n" "$(date +%FT%T)" "$s" "$d" >> $MAN; return; fi
  local n; n=$(stat -c %s "$s"); rm -f "$s" && ln -s "$d" "$s"
  printf "%s\tOK\t%s\t%s\t1\t0\t%s\n" "$(date +%FT%T)" "$s" "$d" "$n" >> $MAN; log "OK $s bytes=$n"
}

df -h /root/autodl-tmp /root/autodl-fs | cat
# finished ZIP archives (dir itself stays local so the archive skill's inside-root check keeps working)
for z in $P/outputs/experiment_archives/*.zip; do move_file "$z" "$DST/outputs/experiment_archives/$(basename "$z")"; done
move_file /root/autodl-tmp/split_dataorigin.zip $DST/misc/split_dataorigin.zip
for r in paper_condseg_f1_30_full paper_screen_pbe_f1 paper_screen_medcagd_f1 paper_screen_condseg_f1; do move_dir $P/runs/$r $DST/runs/$r; done
for o in interactive_refiner_runs stair_validation_20261011 inference_optimization_20261009 unet_continuation_20261012 edg_validation_20261011 overnight_unet_20261012 dino_refiner_cache; do
  move_dir $P/outputs/$o $DST/outputs/$o
done
# relative symlink inside unet_continuation would break after the move -> make it absolute
L=$DST/outputs/unet_continuation_20261012/center_replication
if [ -L "$L" ] && [ ! -e "$L" ]; then ln -sfn $P/outputs/paper_reproduction_20261011/runs/continuation_center_replication_20261012 "$L"; log "fixed relative link $L"; fi
df -h /root/autodl-tmp /root/autodl-fs | cat
log DONE
