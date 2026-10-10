from pathlib import Path
import subprocess
import sys
root=Path(sys.argv[2])
exe=sys.argv[1]
base=[exe,'--model',str(root/'repeat_a.bin'),'--tokenizer',str(root/'tokenizer.bin')]
def run(args,code=0):
    p=subprocess.run(args,text=True,capture_output=True)
    assert p.returncode==code,(args,p.stdout,p.stderr)
    return p
for device in (['cuda'] if '--cuda' in sys.argv[3:] else ['cpu']):
    p=run(base+['--device',device,'--prompt','','--steps','3'])
    assert p.stdout=='aaa\n' and 'generated_tokens=3' in p.stderr,p
for flags in [['--steps','-1'],['--steps','3x'],['--steps','2147483648'],['--temperature','nan'],['--temperature','inf'],['--temperature','-1'],['--seed','-1'],['--device','bad'],['--top-k','-1'],['--typo','abc'],['--prompt','aaaaaaaa']]:
    run(base+flags,1)
run([exe,'--model'],1)
run([exe,'--model','missing','--tokenizer','missing'],1)
assert 'legacy FP32' in run([exe,'--help']).stdout
assert 'Usage' in run([exe]).stdout
print('PASS CLI: CPU/CUDA output, help, malformed arguments and context errors')
