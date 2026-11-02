"""Two-worker job runner. Jobs are read from a text file (one shell command per line, '# done-file' suffix);
a job is skipped when its done-file exists. New lines appended to the file are picked up while running."""
from config import ROOT
import subprocess, time, sys, os, shlex
JOBS = sys.argv[1]; W = int(sys.argv[2]) if len(sys.argv) > 2 else 2
LOG = ROOT + '/work/logs/scheduler_' + os.path.basename(JOBS).split('.')[0] + '.log'
running = {}; started = set()
def log(m):
    open(LOG, 'a').write(time.strftime('%H:%M:%S ') + m + '\n')
while True:
    lines = [l.strip() for l in open(JOBS) if l.strip() and not l.startswith('#')]
    pending = []
    for l in lines:
        cmd, _, done = l.partition(' #DONE ')
        if l in started or (done and os.path.exists(done.strip())):
            continue
        pending.append((l, cmd))
    for p, (l, proc) in list(running.items()):
        if proc.poll() is not None:
            log(f'finished rc={proc.returncode}: {l[:150]}'); del running[p]
    while pending and len(running) < W:
        l, cmd = pending.pop(0); started.add(l)
        proc = subprocess.Popen(cmd, shell=True, cwd=os.path.dirname(os.path.abspath(__file__)))
        running[proc.pid] = (l, proc); log(f'start: {l[:150]}')
    if not running and not pending and os.path.exists(JOBS + '.stop'):
        log('all done'); break
    time.sleep(15)
