#!/usr/bin/env bash
# The vast.ai provider library: everything that is true of this rental service and of
# nothing else -- dual ssh routes, per-machine key refusal, offer filtering, the label-TTL
# contract the reaper reads. A driver talks to a box only through here, so a second
# provider means a second file with these subcommands, not edits scattered through drivers.
#
# Deliberately NOT here: "is the server up, and is it the one I launched". OpenSSH returns
# 255 for any client-side failure anywhere, so that logic is provider-independent and lives
# in lib.sh, where a non-vast experiment will still find it.
#
# Every subcommand is safe to re-run.
#   box.sh rent   <label> <gpu-name> <num-gpus> <disk-gb>   -> prints instance id
#   box.sh wait   <id>                                      -> blocks until ssh answers
#   box.sh ssh    <id> <command...>
#   box.sh sshcmd <id>                                      -> prints the resolved ssh argv
#   box.sh scp    <id> <local> <remote>
#   box.sh tunnel <id> <local-port> <remote-port>
#   box.sh endpoint <id>                                    -> prints the reachable "<host> <port>"
#   box.sh destroy <id>
#   box.sh list
#   box.sh skiplist                                          -> machine ids rent will avoid
set -uo pipefail
IMAGE=${IMAGE:-nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04}
STATE=${STATE:-$(cd "$(dirname "$0")/../.." && pwd)/results/state/instances.log}
# A machine that takes the rental and then refuses the ssh key does it again on the next
# run, and `skip` below only lives for one. Four hours outlasts the fault and is short
# enough that a fixed machine returns on its own. Entries age out; nothing prunes the file.
QUARANTINE=${QUARANTINE:-${STATE%/*}/machine-quarantine.log}
QUARANTINE_TTL=${QUARANTINE_TTL:-14400}
skip_list () {                  # machine ids to avoid: the permanent set plus the fresh quarantine
  { printf '%s\n' "${EXCLUDE_MACHINES:-}" | tr ',' '\n'
    awk -v cut=$(( $(date +%s) - QUARANTINE_TTL )) '$1 > cut {print $2}' "$QUARANTINE" 2>/dev/null
  } | awk '/^[0-9]+$/' | sort -u | paste -sd, -
}
KEY=${VAST_SSH_KEY:-$HOME/.ssh/id_ed25519_vast}
SSH_OPTS=(-i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=no
          -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -o ServerAliveInterval=15)

# Two routes exist and neither works everywhere: some hosts never expose the direct port
# (connect gives "No route to host"), and the vast proxy refuses any key not associated
# with the instance. Probing both is the only way to tell a booting box from a dead one.
candidates () {
  vastai show instance "$1" --raw 2>/dev/null | python3 -c "
import sys, json
try: d = json.load(sys.stdin)
except Exception: raise SystemExit(1)
if d.get('actual_status') != 'running': raise SystemExit(1)
p = (d.get('ports') or {}).get('22/tcp') or []
if d.get('public_ipaddr') and p: print(d['public_ipaddr'], p[0]['HostPort'])
if d.get('ssh_host') and d.get('ssh_port'): print(d['ssh_host'], d['ssh_port'])"
}

# "<status> <status_msg>" -- so a wait can tell a box that is still pulling its image
# from one whose sshd is refusing the key.
instance_state () {
  vastai show instance "$1" --raw 2>/dev/null | python3 -c "
import sys, json
try: d = json.load(sys.stdin)
except Exception: raise SystemExit(0)
print(d.get('actual_status') or '?', (d.get('status_msg') or '').strip().replace('\n', ' ')[:90])"
}

endpoint () {
  local h p
  while read -r h p; do
    [ -n "${h:-}" ] || continue
    ssh "${SSH_OPTS[@]}" -p "$p" "root@$h" true 2>/dev/null && { echo "$h $p"; return 0; }
  done < <(candidates "$1")
  return 1
}

cmd=$1; shift
case "$cmd" in
rent)
  label=$1 gpu=$2 n=$3 disk=$4
  # reaper.sh reads the TTL out of the label. A label without -ttlN falls back to the
  # reaper's generous default, so a short sweep gets the same rope as a long experiment --
  # refuse at the point of renting rather than discover it on the bill.
  case "$label" in
    *-ttl[0-9]*) ;;
    *) echo "label must carry -ttlN so the reaper can bound it: $label" >&2; exit 2 ;;
  esac
  # A rented box that never accepts the ssh key is indistinguishable from a booting one
  # until it has burned the whole wait, and the failure is per-machine: on 2026-08-21 both
  # machines behind one site reported the key attached and refused it, while boxes
  # elsewhere were fine. So reachability is part of renting, not a separate step, and a
  # machine that fails it is excluded from the retry rather than drawn again.
  skip=$(skip_list)
  # Between `create instance` and the handoff below, this shell is the only thing that
  # knows the id: the caller has not been told it yet, so its teardown trap has nothing to
  # destroy. A kill in that window -- up to RENT_CAP*20s of polling -- used to leave a box
  # billing until the reaper's TTL. Own it here, and disown it only once it is printed.
  orphan=
  reap_orphan () {
    [ -n "$orphan" ] || return 0
    echo "rent interrupted; destroying orphan $orphan" >&2
    vastai destroy instance "$orphan" -y >&2
    orphan=
  }
  # A signal trap that only cleans up does not stop the script: bash runs the handler and
  # then RESUMES the loop, so rent would poll on past its own teardown. Each signal handler
  # must exit; only the EXIT trap may fall through.
  trap 'reap_orphan' EXIT
  trap 'reap_orphan; exit 130' INT
  trap 'reap_orphan; exit 143' TERM
  for attempt in $(seq 1 ${RENT_TRIES:-3}); do
    read -r offer machine < <(vastai search offers "gpu_name=\"$gpu\" num_gpus=$n disk_space>=$disk rentable=true" \
            -o dph --raw | RELMIN=${RELMIN:-0.99} NETMIN=${NETMIN:-300} CPUMIN=${CPUMIN:-0} \
            CUDAMIN=${CUDAMIN:-0} DPHMAX=${DPHMAX:-0.22} POWERMIN=${POWERMIN:-0} \
            POWERMAX=${POWERMAX:-0} LANESMIN=${LANESMIN:-0} \
            DLCOSTMAX=${DLCOSTMAX:-0.01} \
            EXCLUDE_MACHINES="$skip" python3 -c "
import sys, json, os
rel, net = float(os.environ['RELMIN']), float(os.environ['NETMIN'])
cpu = float(os.environ['CPUMIN'])
# CUDAMIN guards the prebuilt binary: one built against CUDA 13 dies with 'forward
# compatibility was attempted on non supported HW' on a driver-570 host, and llama.cpp
# then runs on CPU while still printing a CUDA result table.
cuda = float(os.environ['CUDAMIN'])
# A ceiling, because the retry below walks *up* this list: two machines that took the
# rental and then refused the ssh key pushed one arm from the \$0.115 floor to \$0.412,
# and nothing in the loop noticed it had left the price band the arm was budgeted for.
dphmax = float(os.environ['DPHMAX'])
# The power cap is an experiment variable, not a host detail: a 3090 rented at 200 W was
# about to be paired against a 390 W baseline, which measures the cap and not the quant.
# Default 0 so throughput-blind arms are unaffected; a speed leg must set it.
powermin = float(os.environ['POWERMIN'])
# The mirror of POWERMIN, for the leg that WANTS a throttled card: hosts run the same
# model between 110 W and 180 W, which is a power sweep the container cannot perform
# itself (nvidia-smi -pl is refused). 0 means no ceiling.
powermax = float(os.environ['POWERMAX'])
# Tensor split all-reduces across cards once per layer, so lane width is a throughput
# variable for a multi-GPU arm even though it is invisible to a single-card one.
lanesmin = float(os.environ['LANESMIN'])
# Bandwidth is what a leg actually costs, and -o dph cannot see it. Measured on instance
# 49058864: \$0.059 of GPU time, \$0.013 of storage and \$0.711 of download -- 90 percent of
# the bill, on a host whose hourly rate looked competitive. The same pool held offers at
# \$0.0026/GB, a 15x spread. Every leg here pulls a ~16 GiB model, so cap it by default;
# 0 disables the cap for a leg that needs a specific host more than a cheap one.
dlcostmax = float(os.environ['DLCOSTMAX'])
# A floor above the ceiling matches nothing, and the caller then reads the empty result as
# 'the pool has no cheap offer' -- which is what happened the first time a throttled arm
# inherited a driver's 150 W floor. Name the contradiction instead.
if powermax and powermin > powermax:
    sys.exit(f'POWERMIN {powermin:.0f} W is above POWERMAX {powermax:.0f} W: no card can satisfy both')
skip = {int(m) for m in os.environ['EXCLUDE_MACHINES'].split(',') if m.strip()}
d = [o for o in json.load(sys.stdin)
     if (o.get('dph_total') or 1e9) <= dphmax
     and (o.get('gpu_max_power') or 0) >= powermin
     and (powermax == 0 or (o.get('gpu_max_power') or 1e9) <= powermax)
     and (o.get('gpu_lanes') or 0) >= lanesmin
     and (dlcostmax == 0 or (o.get('inet_down_cost') or 0) <= dlcostmax)
     and (o.get('reliability2') or 0) >= rel
     and (o.get('inet_down') or 0) >= net
     and (o.get('cpu_cores_effective') or 0) >= cpu
     and (o.get('cuda_max_good') or 0) >= cuda
     and (o.get('machine_id') not in skip)
     and o.get('verification') == 'verified']
print(f\"{d[0]['id']} {d[0].get('machine_id','')}\" if d else '')")
    [ -n "${offer:-}" ] || { echo "no offer for $gpu x$n under \$${DPHMAX:-0.22}/hr and \$${DLCOSTMAX:-0.01}/GB down (skipping machines: $skip)" >&2; exit 1; }
    # HACK: works around a vast.ai host-daemon bug, not ours. Some hosts hand the
    # container an authorized_keys whose *ownership* violates OpenSSH StrictModes, so sshd
    # refuses a key it was correctly given -- machines at hosts 154152/154620 read as
    # "never accepted the key" until `vastai logs` named it. chmod alone does not fix it;
    # chown does. Looped because the file appears after onstart starts, and bounded so it
    # cannot outlive the rent window. A no-op on healthy hosts, which is why it stays on
    # unconditionally: any machine on the market can then be rented reliably.
    # Remove when vast fixes those hosts; the `vastai logs` capture below is how you find
    # out, and `grep 'bad ownership or modes'` on a failed draw is the test.
    out=$(vastai create instance "$offer" --image "$IMAGE" --disk "$disk" --ssh --direct --label "$label" \
            --onstart-cmd 'for i in $(seq 1 60); do chown -R root:root /root/.ssh 2>/dev/null
                             chmod 700 /root/.ssh 2>/dev/null
                             chmod 600 /root/.ssh/authorized_keys 2>/dev/null
                             sleep 5
                           done' --raw) \
      || { echo "create failed for offer $offer" >&2; exit 1; }
    # Parsed off stdin, not interpolated into source: a quote in the payload would other-
    # wise produce an empty id, and an id we never learn is a bill nothing can stop.
    id=$(printf '%s' "$out" | python3 -c "
import sys, json
try: print(json.load(sys.stdin).get('new_contract') or '')
except Exception: pass")
    [ -n "$id" ] || { echo "create returned no instance id: ${out:0:200}" >&2; exit 1; }
    orphan=$id
    echo "$(date '+%F %T') $id $label $gpu x$n machine=$machine" >> "$STATE"
    # Two separate budgets, because a slow box and a broken box need different patience.
    # RENT_WAIT bounds *reachability* probes and only ticks once the box says `running`, so
    # a cold machine fetching several GB of cudnn-devel cannot be condemned as having
    # "refused the key". RENT_CAP bounds the total wall clock, and 60 (20 min) was under the
    # pull time of a cold 3090 host: it was destroyed mid-`Pulling fs layer`, having been
    # paid for the rental and most of the download, and the retry drew another cold machine
    # to pay again. Overshooting here costs 20s per unused tick; undershooting costs a box.
    probes=0 state='?' msg=''
    for _ in $(seq 1 ${RENT_CAP:-90}); do
      read -r state msg < <(instance_state "$id")
      if [ "$state" = running ]; then
        endpoint "$id" >/dev/null && { orphan=; echo "$id"; exit 0; }
        probes=$((probes + 1))
        [ "$probes" -ge "${RENT_WAIT:-15}" ] && break
      fi
      sleep 20
    done
    if [ "$state" = running ]; then
      echo "$id on machine $machine came up but never accepted the key; destroying and retrying" >&2
    else
      echo "$id on machine $machine never finished starting (status=$state ${msg:+msg=$msg}); destroying and retrying" >&2
    fi
    # The container's own sshd log says why, and it is the only place that does: five
    # rentals were spent guessing at "never accepted the key" before `vastai logs` showed
    # `Authentication refused: bad ownership or modes for file /root/.ssh/authorized_keys`
    # -- the key was delivered and StrictModes rejected it. Capture before destroying.
    vastai logs "$id" 2>&1 | grep -iE 'refused|Failed publickey|Accepted key' | tail -5 >&2
    vastai destroy instance "$id" -y >&2
    orphan=
    [ -n "$machine" ] && printf '%s %s\n' "$(date +%s)" "$machine" >> "$QUARANTINE"
    skip="${skip:+$skip,}$machine"
  done
  echo "no reachable box for $gpu x$n after ${RENT_TRIES:-3} attempts" >&2; exit 1
  ;;
endpoint) endpoint "$1" ;;
wait)
  id=$1
  for _ in $(seq 1 ${WAIT_TRIES:-60}); do
    out=$(endpoint "$id") && { echo "$out"; exit 0; }
    sleep 20
  done
  echo "instance $id never became reachable" >&2; exit 1
  ;;
ssh)
  id=$1; shift
  read -r h p < <(endpoint "$id") || { echo "no reachable endpoint for $id" >&2; exit 1; }
  ssh "${SSH_OPTS[@]}" -p "$p" "root@$h" "$@"
  ;;
sshcmd)
  # Printed rather than executed so a driver can build one runner and reuse it in a hot
  # loop; re-deriving SSH_OPTS in the driver is how the options drifted between files.
  read -r h p < <(endpoint "$1") || { echo "no reachable endpoint for $1" >&2; exit 1; }
  printf '%s\n' ssh "${SSH_OPTS[@]}" -p "$p" "root@$h"
  ;;
scp)
  id=$1 src=$2 dst=$3
  read -r h p < <(endpoint "$id") || { echo "no reachable endpoint for $id" >&2; exit 1; }
  scp "${SSH_OPTS[@]}" -P "$p" "$src" "root@$h:$dst"
  ;;
tunnel)
  id=$1 lport=$2 rport=$3
  read -r h p < <(endpoint "$id") || { echo "no reachable endpoint for $id" >&2; exit 1; }
  # Match the forward spec itself. "ssh -N -L $lport:" cannot match: SSH_OPTS sit between
  # ssh and -N, so that pattern finds nothing whether the tunnel is up or not.
  # Bind one explicit address. A bare "-L $lport:" binds 127.0.0.1 *and* ::1, so a squatter
  # holding only IPv4 lets ssh succeed on IPv6: the forward looks up, ss lists our ssh on
  # the port, and every client still reaches the squatter over 127.0.0.1.
  spec="-L 127.0.0.1:$lport:127.0.0.1:$rport"
  # Supervisors first, then the ssh they own: the other order lets a live supervisor
  # respawn the forward between the two kills.
  for q in $(pgrep -f -- "tunnel-supervise .* $lport $rport" 2>/dev/null); do kill "$q" 2>/dev/null; done
  for q in $(pgrep -f -- "$spec" 2>/dev/null); do kill "$q" 2>/dev/null; done
  sleep 1
  setsid nohup "$0" tunnel-supervise "$h" "$p" "$lport" "$rport" >/dev/null 2>&1 &
  # Ask who owns the port, rather than whether something answers on it. A connect test
  # cannot tell our forward from a squatter: the kill above matches this exact spec, so a
  # tunnel holding the same local port for a different remote survives it, our ssh then
  # dies to ExitOnForwardFailure, and the probe cheerfully measures the squatter's server.
  # Checking liveness first does not fix it either -- our ssh is still alive during its own
  # startup, so the first probe races and passes.
  for _ in $(seq 1 20); do
    for pid in $(ss -ltnpH "sport = :$lport" 2>/dev/null |
                 grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u); do
      tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -qF -- "$spec" && exit 0
    done
    sleep 1
  done
  echo "tunnel $lport->$rport never came up (local port taken by something else?)" >&2
  exit 1
  ;;
tunnel-supervise)
  # Keep one forward alive for as long as the box is. The tunnel drops roughly every 80-90
  # minutes under load and has ended two legs that way; the runner's liveness gate turns
  # that into an abort-and-resume, and this keeps it from reaching the leg at all.
  h=$1 p=$2 lport=$3 rport=$4
  spec="-L 127.0.0.1:$lport:127.0.0.1:$rport"
  child=
  trap 'kill "${child:-0}" 2>/dev/null; exit 0' TERM INT
  fails=0
  while :; do
    ssh "${SSH_OPTS[@]}" -N "$spec" -o ExitOnForwardFailure=yes -p "$p" "root@$h" &
    child=$!
    started=$SECONDS
    wait "$child"
    # A forward that never bound is a different failure from one that served for an hour
    # and dropped. The first means a taken port or a gone box, which retrying cannot fix;
    # the second is the transient this exists for, and must not count toward giving up.
    if [ $((SECONDS - started)) -lt "${TUNNEL_UP_SECONDS:-10}" ]; then
      fails=$((fails + 1))
      [ "$fails" -ge "${TUNNEL_MAX_FAILS:-10}" ] && {
        echo "tunnel $lport->$rport failed $fails times without staying up; giving up" >&2
        exit 1; }
      sleep $(( fails < 5 ? fails : 5 ))
    else
      fails=0
      sleep 1
    fi
  done
  ;;
destroy) vastai destroy instance "$1" -y && echo "destroyed $1" ;;
list)    vastai show instances --raw | python3 -c "
import sys,json,time
for i in json.load(sys.stdin):
    print(i['id'], i.get('label'), i.get('gpu_name'), i.get('actual_status'),
          f\"{(time.time()-(i.get('start_date') or time.time()))/3600:.1f}h\", f\"\${i.get('dph_total',0):.3f}/hr\")" ;;
skiplist) skip_list ;;
*) echo "unknown: $cmd" >&2; exit 2 ;;
esac
