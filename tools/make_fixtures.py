"""Independent NumPy FP64 full-prefix causal reference; no MiniInfer calls."""
from pathlib import Path
import json
import struct
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / 'data' / 'fixtures' / 'framework'
ROOT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20261009)
D, H, L, NH, V, S = 24, 37, 2, 4, 300, 8
TOKENS = np.array([1, 41, 299, 8, 10, 71, 2, 78], dtype='<i4')

def reference(weights, nk):
    # Recompute all visible positions as a batch; no KV cache. Compare each prefix
    # to C++ incremental decoding so a shared cache/indexing bug cannot hide here.
    embedding, an, wq, wk, wv, wo, fn, w1, w2, w3, final, _, cls = weights
    head = D // NH
    def norm(x, w):
        return x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True)+1e-5)*w
    answers = []
    for count in range(1, S+1):
        x = embedding[TOKENS[:count]].astype(np.float64)
        for layer in range(L):
            n = norm(x, an[layer])
            q = (n @ wq[layer].T).reshape(count, NH, head)
            k = (n @ wk[layer].T).reshape(count, nk, head)
            v = (n @ wv[layer].T).reshape(count, nk, head)
            angles = np.arange(count)[:,None,None] / 10000**(np.arange(0,head,2)/head)
            for a in (q,k):
                even, odd = a[:,:,0::2].copy(), a[:,:,1::2].copy()
                a[:,:,0::2] = even*np.cos(angles)-odd*np.sin(angles)
                a[:,:,1::2] = even*np.sin(angles)+odd*np.cos(angles)
            k = np.repeat(k, NH//nk, axis=1)
            v = np.repeat(v, NH//nk, axis=1)
            scores = np.einsum('thd,shd->hts',q,k)/np.sqrt(head)
            scores[:,np.triu_indices(count,1)[0],np.triu_indices(count,1)[1]] = -np.inf
            p = np.exp(scores-np.max(scores,axis=-1,keepdims=True))
            p /= p.sum(axis=-1,keepdims=True)
            attended = np.einsum('hts,shd->thd',p,v).reshape(count,D)
            x = x + attended @ wo[layer].T
            n = norm(x,fn[layer])
            a,b = n @ w1[layer].T, n @ w3[layer].T
            x = x + (a/(1+np.exp(-a))*b) @ w2[layer].T
        answers.append(norm(x[-1], final) @ cls.T)
    return np.array(answers,dtype='<f4')

def create(name,nk,shared,force=None):
    kv = nk*(D//NH)
    def rand(shape): return rng.normal(0,0.17,shape).astype('<f4')
    def norm(shape): return rng.uniform(.7,1.3,shape).astype('<f4')
    emb=rand((V,D)); an=norm((L,D)); fn=norm((L,D)); final=norm((D,))
    q=rand((L,D,D)); k=rand((L,kv,D)); v=rand((L,kv,D)); o=rand((L,D,D))
    w1=rand((L,H,D)); w2=rand((L,D,H)); w3=rand((L,H,D))
    rope=np.zeros((S,D//NH),dtype='<f4')
    cls=emb if shared else rand((V,D))
    if force is not None:
        emb.fill(1); cls.fill(0); cls[force].fill(1)
        an.fill(1); fn.fill(1); final.fill(1)
        for a in (q,k,v,o,w1,w2,w3): a.fill(0)
    arrays=[emb,an,q,k,v,o,fn,w1,w2,w3,final,rope]
    header=struct.pack('<7i',D,H,L,NH,nk,V if shared else -V,S)
    (ROOT/f'{name}.bin').write_bytes(header+b''.join(x.tobytes() for x in arrays)
                                    +(b'' if shared else cls.tobytes()))
    if force is None: reference(arrays+[cls],nk).tofile(ROOT/f'{name}.expected')

names=[]
for nk in [4,2,1]:
    for shared in [True,False]:
        name=f'kv{nk}_{"tied" if shared else "untied"}'
        create(name,nk,shared); names.append(name)
create('eos',2,False,2)
create('bos',2,False,1)
create('repeat_a',2,False,260)
TOKENS.tofile(ROOT/'tokens.bin')

pieces=['<unk>','<s>','</s>']+[f'<0x{i:02X}>' for i in range(256)]
pieces+=[' ','a','b','c','h','e','l','o','he','hel','hell','hello',' hello','你','好','你好',' 你好']
while len(pieces)<V: pieces.append(f'<unused_{len(pieces)}>')
encoded=[p.encode() for p in pieces]
tokenizer=struct.pack('<I',max(map(len,encoded)))
for i,p in enumerate(encoded):
    tokenizer+=struct.pack('<fi',float(i),len(p))+p
(ROOT/'tokenizer.bin').write_bytes(tokenizer)
(ROOT/'tokenizer_short.bin').write_bytes(tokenizer[:-1])
bad=bytearray(tokenizer); bad[8:12]=struct.pack('<i',2**31-1)
(ROOT/'tokenizer_bad_length.bin').write_bytes(bad)
source=(ROOT/'kv2_untied.bin').read_bytes()
(ROOT/'truncated.bin').write_bytes(source[:-4])
(ROOT/'trailing.bin').write_bytes(source+b'xxxx')
for name,index,value in [('bad_vocab',5,-2**31),('bad_heads',3,5),('bad_seq',6,0),('bad_rope',3,8),('bad_size',2,2**31-1)]:
    bad=bytearray(source); bad[index*4:index*4+4]=struct.pack('<i',value)
    (ROOT/f'{name}.bin').write_bytes(bad)
(ROOT/'manifest.json').write_text(json.dumps({'models':names,'reference':'NumPy FP64 batched causal, all prefixes','seed':20261009},indent=2))
print('Created 6 numeric fixtures, 3 generation fixtures and malformed files')
