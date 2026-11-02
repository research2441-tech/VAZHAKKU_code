"""Stateless job runner. Jobs come from one or more text files (one shell command per line, ' #DONE file' suffix).
A job is done when its done-file exists and running when a process whose command line contains the done-file stem
is alive, so the runner can be restarted at any time without duplicating work. Children run in their own session."""
from config import ROOT
import subprocess, time, sys, os
W = int(sys.argv[1]); FILES = sys.argv[2:]
LOG = ROOT + '/work/logs/scheduler2.log'
def log(m): open(LOG, 'a').write(time.strftime('%H:%M:%S ') + m + '\n')
def alive_cmds():
    out = subprocess.run(['ps', '-eo', 'args'], capture_output=True, text=True).stdout.split('\n')
    return [l for l in out if l.startswith('python3 bridge.py') or l.startswith('python3 train_dialect.py')]
def key(done):
    return os.path.basename(done).rsplit('.', 1)[0]          # e.g. bmargin_hard_s0, hyper_grl_s0
log('start W=%d %s' % (W, FILES))
while True:
    jobs = []
    for f in FILES:
        for l in open(f):
            l = l.strip()
            if l and not l.startswith('#') and ' #DONE ' in l:
                cmd, _, done = l.partition(' #DONE '); jobs.append((cmd, done.strip()))
    cmds = alive_cmds()
    running = [k for c, d in jobs if not os.path.exists(d) for k in [key(d)] if any(k in a for a in cmds)]
    pending = [(c, d) for c, d in jobs if not os.path.exists(d) and key(d) not in running]
    if not running and not pending:
        log('all done'); break
    while pending and len(running) < W:
        c, d = pending.pop(0)
        subprocess.Popen(c, shell=True, cwd=os.path.dirname(os.path.abspath(__file__)), start_new_session=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        running.append(key(d)); log('start ' + key(d))
    time.sleep(20)
