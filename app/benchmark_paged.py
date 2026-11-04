#!/usr/bin/env python3
"""Independent end-to-end legacy versus direct-I/O paged model benchmark.

Only Python's standard library is required. Cold means verified Linux guest
page-cache residency when mincore is available, never physical-NVMe coldness.
"""
import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import struct
import subprocess
import time

FIELDS=('dim','hidden_dim','layer_num','head_num','kv_head_num','vocab_size','seq_len')

def layout(config):
    d,h,l,q,k,v,n=(config[key] for key in FIELDS)
    if min(d,h,l,q,k,n)<=0 or not v or d%q or q%k or (d//q)%2:
        raise ValueError('Invalid legacy FP32 model geometry')
    entries=[]
    cursor=28
    def add(name,shape):
        nonlocal cursor
        size=math.prod(shape)*4
        entries.append(dict(name=name,shape=shape,offset=cursor,bytes=size))
        cursor+=size
    add('tok_embeddings.weight',[abs(v),d])
    for i in range(l):add(f'layers.{i}.attention_norm.weight',[d])
    for name,rows in (('wq',d),('wk',k*(d//q)),('wv',k*(d//q)),('wo',d)):
        for i in range(l):add(f'layers.{i}.attention.{name}.weight',[rows,d])
    for i in range(l):add(f'layers.{i}.ffn_norm.weight',[d])
    for name,shape in (('w1',[h,d]),('w2',[d,h]),('w3',[h,d])):
        for i in range(l):add(f'layers.{i}.feed_forward.{name}.weight',shape)
    add('norm.weight',[d])
    cursor+=n*(d//q)*4  # two legacy RoPE tables, unused by native forward
    if v<0:add('output.weight',[abs(v),d])
    return entries,cursor

def generate_legacy(path,config,seed=20261103):
    entries,size=layout(config)
    rng=random.Random(seed)
    path=Path(path)
    with path.open('xb') as f:
        f.write(struct.pack('<7i',*(config[k] for k in FIELDS)))
        for entry in entries:
            if f.tell()!=entry['offset']:f.write(b'\0'*(entry['offset']-f.tell()))
            count=entry['bytes']//4
            for start in range(0,count,65536):
                amount=min(65536,count-start)
                norm='norm' in entry['name']
                scale=.4 if entry['name']=='tok_embeddings.weight' else .25/math.sqrt(config['dim'])
                values=[1+rng.uniform(-.05,.05) if norm else rng.uniform(-scale,scale) for _ in range(amount)]
                f.write(struct.pack('<'+'f'*amount,*values))
        if f.tell()<size:f.write(b'\0'*(size-f.tell()))
        f.flush();os.fsync(f.fileno())

def convert_legacy(source,destination):
    source,destination=Path(source),Path(destination)
    with source.open('rb') as f:
        raw=f.read(28)
        if len(raw)!=28:raise ValueError('Truncated legacy header')
        config=dict(zip(FIELDS,struct.unpack('<7i',raw)))
        entries,expected=layout(config)
        if source.stat().st_size!=expected:raise ValueError('Expected a complete legacy FP32 v0 checkpoint')
        metadata={f'miniinfer.{k}':str(abs(v) if k=='vocab_size' else v) for k,v in config.items()}
        metadata['miniinfer.layout']='llama2c-f32-v0'
        header={'__metadata__':metadata}
        cursor=0
        for entry in entries:
            header[entry['name']]={'dtype':'F32','shape':entry['shape'],'data_offsets':[cursor,cursor+entry['bytes']]}
            cursor+=entry['bytes']
        raw=json.dumps(header,separators=(',',':')).encode()
        raw+=b' '*((-len(raw)-8)%4096)  # align data start; no holes inside the tensor buffer
        digest=hashlib.sha256()
        with destination.open('xb') as target:
            target.write(struct.pack('<Q',len(raw)));target.write(raw)
            for entry in entries:
                f.seek(entry['offset']);remaining=entry['bytes']
                while remaining:
                    block=f.read(min(1<<20,remaining))
                    if not block:raise ValueError('Truncated source tensor')
                    target.write(block);digest.update(block);remaining-=len(block)
            target.flush();os.fsync(target.fileno())
    return dict(config=config,canonical_payload_sha256=digest.hexdigest(),tensor_count=len(entries))

def verify_pair(source,destination):
    with Path(source).open('rb') as a,Path(destination).open('rb') as b:
        config=dict(zip(FIELDS,struct.unpack('<7i',a.read(28))))
        entries,expected=layout(config)
        if Path(source).stat().st_size!=expected:raise ValueError('Legacy length mismatch')
        length=struct.unpack('<Q',b.read(8))[0]
        if length>100000000:raise ValueError('Safetensors header exceeds limit')
        header=json.loads(b.read(length));first,second=hashlib.sha256(),hashlib.sha256()
        for entry in entries:
            tensor=header[entry['name']]
            if tensor['dtype']!='F32' or tensor['shape']!=entry['shape']:raise ValueError('Tensor metadata mismatch')
            begin,end=tensor['data_offsets']
            if end-begin!=entry['bytes']:raise ValueError('Tensor length mismatch')
            a.seek(entry['offset']);b.seek(8+length+begin);remaining=entry['bytes']
            while remaining:
                n=min(1<<20,remaining);x,y=a.read(n),b.read(n)
                if len(x)!=n or x!=y:raise ValueError('Paired checkpoints contain different tensor bytes')
                first.update(x);second.update(y);remaining-=n
        return dict(config=config,canonical_payload_sha256=first.hexdigest(),safetensors_payload_sha256=second.hexdigest(),tensor_count=len(entries))

def residency(path):
    """Inspect residency without faulting file data into the cache."""
    libc=ctypes.CDLL(None,use_errno=True)
    libc.mmap.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_longlong]
    libc.mmap.restype=ctypes.c_void_p
    libc.mincore.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p]
    libc.munmap.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
    fd=os.open(path,os.O_RDONLY)
    size=os.fstat(fd).st_size
    try:
        address=libc.mmap(None,size,1,1,fd,0)
        if address==ctypes.c_void_p(-1).value:return None
        try:
            count=(size+os.sysconf('SC_PAGE_SIZE')-1)//os.sysconf('SC_PAGE_SIZE')
            pages=(ctypes.c_ubyte*count)()
            if libc.mincore(address,size,pages):return None
            return dict(resident_pages=sum(bool(x&1) for x in pages),total_pages=count)
        finally:libc.munmap(address,size)
    finally:os.close(fd)

def cache_condition(path,mode):
    if mode=='warm':
        with path.open('rb') as f:
            while f.read(1<<20):pass
    else:
        fd=os.open(path,os.O_RDONLY)
        try:os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
        finally:os.close(fd)
    result=residency(path)
    return dict(requested=mode,residency_before=result,
                guest_cold_verified=mode=='guest-cold' and result is not None and result['resident_pages']==0,
                physical_nvme_cold_verified=False)

def quantile(values,q):
    values=sorted(values);offset=(len(values)-1)*q;i=int(offset)
    return values[i]+(values[min(i+1,len(values)-1)]-values[i])*(offset-i)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--legacy',type=Path)
    parser.add_argument('--safetensors',type=Path)
    parser.add_argument('--generate-fixture',action='store_true')
    parser.add_argument('--rounds',type=int,default=3)
    parser.add_argument('--prompt-tokens',type=int,default=48)
    parser.add_argument('--generate-tokens',type=int,default=12)
    parser.add_argument('--page-tokens',type=int,default=8)
    parser.add_argument('--gpu-kv-bytes',type=int,default=1<<20)
    parser.add_argument('--host-kv-bytes',type=int,default=512<<10)
    parser.add_argument('--cache',choices=['guest-cold','warm'],default='guest-cold')
    parser.add_argument('--include-demand-only',action='store_true')
    args=parser.parse_args()
    if min(args.rounds,args.prompt_tokens,args.generate_tokens,args.page_tokens,args.gpu_kv_bytes,args.host_kv_bytes)<1:
        parser.error('Positive counts and budgets required')
    args.output.mkdir(parents=True,exist_ok=False)
    legacy=args.legacy.resolve() if args.legacy else args.output/'model.bin'
    if args.generate_fixture:
        if args.legacy:parser.error('Use --legacy or --generate-fixture')
        generate_legacy(legacy,dict(dim=64,hidden_dim=128,layer_num=4,head_num=8,kv_head_num=4,vocab_size=-128,seq_len=max(128,args.prompt_tokens+args.generate_tokens+16)))
    if not legacy.is_file():parser.error('Provide --legacy or --generate-fixture')
    safetensors=args.safetensors.resolve() if args.safetensors else args.output/'model.safetensors'
    if not args.safetensors:convert_legacy(legacy,safetensors)
    pair=verify_pair(legacy,safetensors)
    if args.prompt_tokens+args.generate_tokens>pair['config']['seq_len']:parser.error('Prompt and generation exceed context')
    affinity=sorted(os.sched_getaffinity(0))[:4]
    variants=['legacy','paged-resident','paged-tiered']+(['paged-demand'] if args.include_demand_only else [])
    manifest=dict(binary=str(args.binary.resolve()),binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                  pair=pair,affinity=affinity,arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  baseline='mmap legacy FP32 loader + contiguous CUDA KV',
                  candidate='standard named FP32 Safetensors io_uring+O_DIRECT + PagedKV',
                  root_cmake_sha256=hashlib.sha256((Path(__file__).resolve().parents[1]/'CMakeLists.txt').read_bytes()).hexdigest(),
                  kernel=os.uname().release,source_sha256={str(p.relative_to(Path(__file__).resolve().parents[1])):hashlib.sha256(p.read_bytes()).hexdigest() for directory in ('include','src','cmake','app') for p in (Path(__file__).resolve().parents[1]/directory).rglob('*') if p.is_file() and p.suffix in ('.h','.cpp','.cu','.cuh','.cmake','.py')},
                  platform_note='Cache residency covers Linux guest pages only; no physical cold-NVMe claim')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    commands=[];results=[];expected=None;expected_hash=None
    for trial in range(args.rounds):
        order=variants[trial%len(variants):]+variants[:trial%len(variants)]
        for variant in order:
            model=legacy if variant=='legacy' else safetensors
            condition=cache_condition(model,args.cache)
            command=['taskset','-c',','.join(map(str,affinity)),str(args.binary.resolve()),
                     '--model',str(model),'--variant',variant,'--prompt',str(args.prompt_tokens),
                     '--generate',str(args.generate_tokens),'--page-tokens',str(args.page_tokens),
                     '--gpu-bytes',str(args.gpu_kv_bytes),'--host-bytes',str(args.host_kv_bytes)]
            start=time.monotonic()
            run=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            stem=f'{trial}-{variant}'
            (args.output/(stem+'.stderr.log')).write_text(run.stderr)
            commands.append(dict(command=command,returncode=run.returncode,cache=condition,process_wall_ms=(time.monotonic()-start)*1000))
            (args.output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
            if run.returncode:raise RuntimeError(f'{stem} failed: {run.stderr[-2000:]}')
            result=json.loads(run.stdout);result.update(trial=trial,cache=condition,process_wall_ms=commands[-1]['process_wall_ms'])
            if result['miss']['tokens']!=result['hit']['tokens'] or not result['prefix_logits_bitwise_equal']:
                raise RuntimeError('Prefix reuse changed logits or generated tokens')
            if expected is None:expected=result['miss']['tokens'];expected_hash=result['miss']['first_logit_hash']
            if result['miss']['first_logit_hash']!=expected_hash:raise RuntimeError('First logits hash differs across variants')
            if result['miss']['tokens']!=expected:raise RuntimeError('Variants produced different tokens')
            if variant!='legacy' and result['weight_backend']!='io_uring+O_DIRECT+pinned':
                raise RuntimeError('Candidate did not use requested loader')
            if variant.startswith('paged') and result['hit']['stats']['last_prefix_hit_tokens']==0:
                raise RuntimeError('Prompt produced no full-page prefix hit; increase its length')
            if variant in ('paged-tiered','paged-demand'):
                usage=result['hit']['stats']
                if min(usage['ssd_read_bytes'],usage['ssd_written_bytes'],usage['uring_submissions'])==0:
                    raise RuntimeError('Tiered variant did not exercise SSD io_uring; reduce KV budgets')
                if variant=='paged-tiered' and usage['prefetched_pages']==0:
                    raise RuntimeError('Tiered variant did not exercise layer prefetch')
                if variant=='paged-demand' and usage['prefetched_pages']!=0:
                    raise RuntimeError('Demand-only variant unexpectedly prefetched')
            (args.output/(stem+'.json')).write_text(json.dumps(result,indent=2)+'\n');results.append(result)
            print(stem,'verified',round(result['model_init_ms'],3),round(result['miss']['ttft_ms'],3),round(result['hit']['ttft_ms'],3),flush=True)
    summary=[]
    for variant in variants:
        group=[r for r in results if r['variant']==variant]
        row=dict(variant=variant,processes=len(group),guest_cold_verified_processes=sum(r['cache']['guest_cold_verified'] for r in group),weight_backend=group[0]['weight_backend'])
        for key in ('context_init_ms','model_init_ms','startup_through_first_token_ms','process_wall_ms'):
            row[key]=statistics.median(r[key] for r in group)
        for request in ('miss','hit'):
            row[request+'_ttft_ms']=statistics.median(r[request]['ttft_ms'] for r in group)
            row[request+'_decode_tokens_s']=statistics.median(r[request]['decode_tokens_s'] for r in group)
            row[request+'_request_ms']=statistics.median(r[request]['request_ms'] for r in group)
            row[request+'_end_to_end_output_tokens_s']=statistics.median(r[request]['end_to_end_output_tokens_s'] for r in group)
            samples=[x for r in group for x in r[request]['decode_samples_ms']]
            row[request+'_tpot_p50_ms']=statistics.median(samples) if samples else None
            row[request+'_tpot_p99_ms']=quantile(samples,.99) if samples else None
        row['prefix_hit_tokens']=group[0]['hit']['stats']['last_prefix_hit_tokens']
        row['hit_prefill_ssd_read_bytes']=statistics.median(r['hit']['stats_after_prefill']['ssd_read_bytes']-r['hit']['stats_before']['ssd_read_bytes'] for r in group)
        for key in ('ssd_read_bytes','ssd_written_bytes','uring_submissions','gpu_peak_bytes','host_peak_bytes','ssd_peak_bytes'):
            row[key]=None if variant=='legacy' and key.endswith('_peak_bytes') else statistics.median(r['hit']['stats'][key] for r in group)
        summary.append(row)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (args.output/'complete.json').write_text(json.dumps(dict(complete=True,processes=len(results),tokens_equal=True,prefix_logits_bitwise_equal=True))+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
